"""Materialize model-owned mutable workspaces before tensor bufferization.

Numerical kernels remain external. Only descriptors and standard buffer/tensor
boundary operations are introduced; no candidate arithmetic is generated.
"""

from __future__ import annotations


def unique_subgraph_inputs(driver):
    """Normalize occurrence-based input discovery in older Buddy releases.

    A root is one caller-owned allocation even when many primitive calls use
    it. This is model graph plumbing, and deliberately does not alter Buddy.
    """
    from buddy.compiler.graph.operation import PlaceholderOp

    for name, subgraph in driver._subgraphs.items():
        roots = list(dict.fromkeys(driver._subgraphs_inputs[name]))
        placeholders = {
            node.name: node for node in subgraph.body if isinstance(node, PlaceholderOp)
        }
        body = [placeholders[root] for root in roots]
        body.extend(
            node for node in subgraph.body if not isinstance(node, PlaceholderOp)
        )
        subgraph._body = body
        subgraph._inputs = list(range(len(roots)))
        subgraph.node_table = {node.name: node for node in body}
        driver._subgraphs_inputs[name] = roots


def normalize_wrapper_workspace_abi(source, callee, argument_roots, workspace_names):
    """Keep the wrapper's workspace call ABI equal to the mutable subgraph."""
    from buddy_mlir import ir

    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(source)
        functions = {
            ir.StringAttr(op.attributes["sym_name"]).value: op
            for op in module.body.operations
            if op.operation.name == "func.func"
        }
        declaration = functions[callee]
        original = ir.FunctionType(
            ir.TypeAttr(declaration.attributes["function_type"]).value
        )
        if len(original.inputs) != len(argument_roots):
            raise ValueError("wrapper subgraph argument mapping differs")
        inputs = list(original.inputs)
        positions = [
            index
            for index, root in enumerate(argument_roots)
            if root in workspace_names
        ]
        for index in positions:
            old = ir.MemRefType(inputs[index])
            inputs[index] = ir.MemRefType.get(old.shape, old.element_type)
        declaration.attributes["function_type"] = ir.TypeAttr.get(
            ir.FunctionType.get(inputs, original.results)
        )
        matched = 0
        for function in functions.values():
            if not function.regions[0].blocks:
                continue
            for op in function.regions[0].blocks[0].operations:
                if (
                    op.operation.name != "func.call"
                    or ir.FlatSymbolRefAttr(op.attributes["callee"]).value != callee
                ):
                    continue
                matched += 1
                for index in positions:
                    value = op.operands[index]
                    while (
                        hasattr(value.owner, "operation")
                        and value.owner.operation.name == "memref.cast"
                    ):
                        value = value.owner.operands[0]
                    if value.type != inputs[index]:
                        raise ValueError(
                            "wrapper workspace must originate in a static dense zero-offset caller memref"
                        )
                    op.operands[index] = value
        if matched != 1:
            raise ValueError("expected one wrapper subgraph invocation")
        module.operation.verify()
        return str(module), {
            "status": "PASS",
            "callee": callee,
            "mutable_workspace_positions": positions,
            "static_dense_zero_offset": True,
            "scope": "typed wrapper descriptor ABI; no numerical execution",
        }


def materialize_workspace_buffers(
    source, function_name, argument_roots, workspace_names, calls
):
    from buddy_mlir import ir

    targets = {call["symbol"] for call in calls if call.get("void")}
    if len(targets) != sum(call.get("void", False) for call in calls):
        raise ValueError("duplicate external workspace call symbol")
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(source)
        functions = {
            ir.StringAttr(op.attributes["sym_name"]).value: op
            for op in module.body.operations
            if op.operation.name == "func.func"
        }
        function = functions[function_name]
        block = function.regions[0].blocks[0]
        if len(block.arguments) != len(argument_roots):
            raise ValueError(
                "workspace argument mapping does not match captured function"
            )
        if len(argument_roots) != len(set(argument_roots)):
            raise ValueError(
                "caller allocation roots must be unique function arguments"
            )
        order = {
            operation.operation: index
            for index, operation in enumerate(block.operations)
        }
        call_by_symbol = {call["symbol"]: call for call in calls if call.get("void")}
        materialized = []
        for index, (argument, root) in enumerate(zip(block.arguments, argument_roots)):
            if root not in workspace_names:
                continue
            original_type = ir.RankedTensorType(argument.type)
            buffer_type = ir.MemRefType.get(
                original_type.shape, original_type.element_type
            )
            tensor_uses = []
            for use in list(argument.uses):
                owner = use.owner
                callee = (
                    ir.FlatSymbolRefAttr(owner.attributes["callee"]).value
                    if owner.name == "func.call"
                    else None
                )
                if callee not in targets:
                    tensor_uses.append(use)
            argument.set_type(buffer_type)
            if tensor_uses:
                if any(use.owner not in order for use in tensor_uses):
                    raise ValueError(
                        "workspace tensor view must be in the captured entry block"
                    )
                first = min(tensor_uses, key=lambda use: order[use.owner]).owner
                writers = [
                    operation
                    for operation in block.operations
                    if operation.operation.name == "func.call"
                    and ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
                    in targets
                    and any(
                        argument == operation.operands[position]
                        for position in call_by_symbol[
                            ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
                        ].get("destination_indices", [])
                    )
                ]
                if (
                    not writers
                    or max(order[operation.operation] for operation in writers)
                    >= order[first]
                ):
                    raise ValueError(
                        f"workspace {root} tensor consumer must follow its final primitive writer"
                    )
                view = ir.Operation.create(
                    "bufferization.to_tensor",
                    results=[original_type],
                    operands=[argument],
                    attributes={
                        "restrict": ir.UnitAttr.get(),
                        "writable": ir.UnitAttr.get(),
                    },
                    ip=ir.InsertionPoint(first),
                )
                for use in tensor_uses:
                    use.owner.operands[use.operand_number] = view.results[0]
            materialized.append(
                {
                    "argument": index,
                    "root": root,
                    "tensor_consumers": len(tensor_uses),
                    "tensor_views": int(bool(tensor_uses)),
                    "restrict_unique_root": True,
                }
            )
        original_function_type = ir.FunctionType(
            ir.TypeAttr(function.attributes["function_type"]).value
        )
        function.attributes["function_type"] = ir.TypeAttr.get(
            ir.FunctionType.get(
                [argument.type for argument in block.arguments],
                original_function_type.results,
            )
        )
        seen = []
        for operation in list(block.operations):
            if operation.operation.name != "func.call":
                continue
            callee = ir.FlatSymbolRefAttr(operation.attributes["callee"]).value
            if callee not in targets:
                continue
            if operation.results:
                raise ValueError(
                    "mutable workspace calls must have explicit void destinations"
                )
            call = call_by_symbol[callee]
            for index, value in enumerate(list(operation.operands)):
                if isinstance(value.type, ir.RankedTensorType):
                    if index in call.get("destination_indices", []):
                        raise ValueError(
                            "primitive destination must be caller-owned mutable memref"
                        )
                    tensor_type = ir.RankedTensorType(value.type)
                    root = (
                        call["operands"][index]
                        if isinstance(call["operands"][index], str)
                        else call["operands"][index].name
                    )
                    if hasattr(value.owner, "operation"):
                        value.owner.operation.location = ir.Location.name(
                            "model_value:" + root
                        )
                    buffer_type = ir.MemRefType.get(
                        tensor_type.shape, tensor_type.element_type
                    )
                    view = ir.Operation.create(
                        "bufferization.to_buffer",
                        results=[buffer_type],
                        operands=[value],
                        attributes={"read_only": ir.UnitAttr.get()},
                        ip=ir.InsertionPoint(operation),
                        loc=ir.Location.name("model_readonly:" + root),
                    )
                    operation.operands[index] = view.results[0]
                elif not isinstance(value.type, ir.MemRefType):
                    raise ValueError(
                        "external workspace operand must be a tensor or memref"
                    )
            declaration = functions[callee]
            declaration.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get([value.type for value in operation.operands], [])
            )
            seen.append(callee)
        expected = [call["symbol"] for call in calls if call.get("void")]
        if seen != expected:
            raise ValueError("workspace materialization changed external call order")
        module.operation.verify()
        return module.operation.get_asm(enable_debug_info=True), {
            "schema": 1,
            "status": "PASS",
            "function": function_name,
            "argument_roots": list(argument_roots),
            "workspace_roots": sorted(workspace_names),
            "mutable_workspace_arguments": materialized,
            "actual_void_call_symbols": seen,
            "scope": "typed workspace memref ABI and readonly tensor-to-buffer bridges; no numerical execution",
        }


def verify_bufferized_workspace_dataflow(
    source, function_name, argument_roots, workspace_names, calls
):
    """Audit real bufferized SSA roots, including the only allowed copies.

    Workspace destinations/readers must resolve directly to the exact caller
    argument. Copy-initialized allocations may only be written by their own
    initializing copies and read by captured calls or pure plumbing transposes;
    a tensor-materialization allocation completed by several copies is accepted
    only when every writer is one of those copies and no caller workspace root
    feeds it. A copy-initialized allocation can never be a primitive
    destination.
    """
    import re
    from buddy_mlir import ir

    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(source)
        function = next(
            op
            for op in module.body.operations
            if op.operation.name == "func.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == function_name
        )
        block = function.regions[0].blocks[0]
        if len(block.arguments) != len(argument_roots):
            raise ValueError(
                "bufferized function arguments differ from the captured ABI"
            )
        arguments = {
            value: (index, root)
            for index, (value, root) in enumerate(zip(block.arguments, argument_roots))
        }
        aliases = {
            "memref.cast",
            "memref.collapse_shape",
            "memref.expand_shape",
            "memref.subview",
            "memref.reinterpret_cast",
            "memref.memory_space_cast",
        }

        def walk(operation):
            yield operation
            for region in operation.regions:
                for inner_block in region.blocks:
                    for inner in inner_block.operations:
                        yield from walk(inner.operation)

        operations = list(walk(function.operation))
        operation_order = {
            operation: index for index, operation in enumerate(operations)
        }
        copies = [op for op in operations if op.name == "memref.copy"]
        targets = {call["symbol"]: call for call in calls if call.get("void")}
        captured_calls = {call["symbol"]: call for call in calls}
        actual_calls = [
            op
            for op in operations
            if op.name == "func.call"
            and ir.FlatSymbolRefAttr(op.attributes["callee"]).value in targets
        ]
        if [
            ir.FlatSymbolRefAttr(op.attributes["callee"]).value for op in actual_calls
        ] != list(targets):
            raise ValueError("bufferized primitive call order differs from capture")

        def alias_base(value):
            while (
                value not in arguments
                and hasattr(value.owner, "operation")
                and value.owner.operation.name in aliases
            ):
                value = value.owner.operands[0]
            return value

        destinations = {
            alias_base(op.operands[index])
            for op in actual_calls
            for index in targets[ir.FlatSymbolRefAttr(op.attributes["callee"]).value][
                "destination_indices"
            ]
        }
        copy_into = {}
        for op in copies:
            copy_into.setdefault(alias_base(op.operands[1]), []).append(op)

        def audit_readonly_uses(alias, checked, initializers):
            """Copied buffers may not be written after initialization.

            Only the initializing copies may write the allocation; captured
            calls must read it (never as a destination) and must run after the
            final initializer.  Pure model plumbing may consume it read-only:
            the bufferized transitive lowerers emit linalg.transpose, whose
            operand 0 is the read side and operand 1 its own fresh output.
            """
            if alias in checked:
                return
            checked.add(alias)
            last_writer = max(
                operation_order[initializer] for initializer in initializers
            )
            for use in alias.uses:
                user = use.owner.operation
                if user.name in aliases:
                    for result in user.results:
                        audit_readonly_uses(result, checked, initializers)
                elif user.name == "func.call":
                    symbol = ir.FlatSymbolRefAttr(user.attributes["callee"]).value
                    contract = captured_calls.get(symbol)
                    if contract is None or use.operand_number in contract.get(
                        "destination_indices", []
                    ):
                        raise ValueError(
                            "readonly layout allocation is passed to an unknown/writing call"
                        )
                    if operation_order[user] <= last_writer:
                        raise ValueError(
                            "readonly layout allocation is read before its initializing copy"
                        )
                elif user.name == "memref.copy":
                    if use.operand_number == 1 and user not in initializers:
                        raise ValueError(
                            "readonly layout allocation has a second copy writer"
                        )
                elif user.name == "linalg.transpose":
                    if use.operand_number != 0:
                        raise ValueError(
                            "readonly layout allocation has an unapproved writer/use"
                        )
                elif user.name not in {"memref.dealloc", "memref.dim"}:
                    raise ValueError(
                        "readonly layout allocation has an unapproved writer/use"
                    )

        def origin(value, follow_copy=True, visited=None):
            visited = set() if visited is None else visited
            if value in visited:
                raise ValueError("cyclic buffer alias")
            visited.add(value)
            if value in arguments:
                index, root = arguments[value]
                return {"base_argument": index, "base_root": root, "aliases": []}
            owner = value.owner
            if not hasattr(owner, "operation"):
                raise ValueError("unexpected region block argument in primitive buffer")
            owner = owner.operation
            result = None
            if owner.name in aliases:
                result = origin(owner.operands[0], follow_copy, visited)
            elif follow_copy and value in copy_into:
                incoming = copy_into[value]
                if (
                    owner.name != "memref.alloc"
                    or len(incoming) != 1
                    or value in destinations
                ):
                    raise ValueError(
                        "readonly layout allocation has multiple writers or is a primitive destination"
                    )
                # A fresh allocation's only write may be its initializing copy.
                audit_readonly_uses(value, set(), incoming)
                result = origin(incoming[0].operands[0], follow_copy, visited)
                if result.get("base_root") in workspace_names:
                    raise ValueError(
                        "workspace was copied into a readonly layout allocation"
                    )
                result = {**result, "readonly_copy": True}
            else:
                result = {
                    "producer": owner.name,
                    "producer_location": str(owner.location),
                    "aliases": [],
                }
            result = {**result, "aliases": [owner.name] + result.get("aliases", [])}
            tagged = re.search(r'model_value:([^"\s)]+)', str(owner.location))
            if tagged:
                result["logical_root"] = tagged[1]
            return result

        records = []
        workspace_operands = destination_count = 0
        for op in actual_calls:
            symbol = ir.FlatSymbolRefAttr(op.attributes["callee"]).value
            call = targets[symbol]
            operands, details = [], []
            for index, value in enumerate(op.operands):
                expected = call["operands"][index]
                expected = expected if isinstance(expected, str) else expected.name
                detail = origin(value)
                if expected in workspace_names:
                    if detail.get("base_root") != expected or detail.get(
                        "readonly_copy"
                    ):
                        raise ValueError(
                            f"workspace reader/writer disconnected: {symbol} operand {index}"
                        )
                    actual = detail["base_root"]
                    workspace_operands += 1
                else:
                    actual = detail.get("logical_root", detail.get("base_root"))
                    if actual != expected:
                        raise ValueError(
                            f"readonly activation root differs: {symbol} operand {index}: {actual} != {expected}"
                        )
                if index in call["destination_indices"]:
                    if (
                        expected not in workspace_names
                        or detail.get("base_root") != expected
                    ):
                        raise ValueError(
                            "primitive destination is not its caller-owned workspace"
                        )
                    destination_count += 1
                operands.append(actual)
                details.append(detail)
            records.append(
                {"symbol": symbol, "operands": operands, "dataflow": details}
            )
        provenance_cache = {}

        def provenance_roots(value, active=None):
            """Caller roots that can feed a copied buffer.

            Follow aliases, copy-initialized local allocations and the pure
            model plumbing transposes one-shot-bufferization emits.  A caller
            workspace root in this set means mutable model data was smuggled
            through a hidden copy.  Opaque producers (compiled calls) cannot
            be caller roots.
            """
            if value in provenance_cache:
                return provenance_cache[value]
            active = set() if active is None else active
            if value in active:
                raise ValueError("cyclic buffer alias")
            active.add(value)
            if value in arguments:
                roots = {arguments[value][1]}
            else:
                owner = value.owner
                if not hasattr(owner, "operation"):
                    raise ValueError(
                        "unexpected region block argument in primitive buffer"
                    )
                owner = owner.operation
                if owner.name in aliases:
                    roots = provenance_roots(owner.operands[0], active)
                elif owner.name == "memref.alloc":
                    roots = set()
                    for writer in copy_into.get(value, []):
                        roots |= provenance_roots(writer.operands[0], active)
                    for use in value.uses:
                        user = use.owner.operation
                        if user.name == "linalg.transpose" and use.operand_number == 1:
                            roots |= provenance_roots(user.operands[0], active)
                elif owner.name == "linalg.transpose":
                    roots = provenance_roots(owner.operands[0], active)
                else:
                    roots = set()
            active.discard(value)
            provenance_cache[value] = roots
            return roots

        readonly_copies = []
        materialized_copies = []
        checked_allocs = set()
        for copy in copies:
            destination = alias_base(copy.operands[1])
            if destination in arguments:
                if arguments[destination][1] in workspace_names:
                    raise ValueError("bufferization inserted a workspace copy")
                continue
            if destination.owner.operation.name != "memref.alloc":
                raise ValueError("copy destination is not a local allocation")
            if destination in destinations:
                raise ValueError(
                    "readonly layout allocation is a primitive destination"
                )
            incoming = copy_into.get(destination, [])
            if destination in checked_allocs:
                continue
            checked_allocs.add(destination)
            # Every writer of a copy-initialized allocation must be one of its
            # own initializing copies; call destinations and unknown readers
            # are still rejected by the shared audit.
            audit_readonly_uses(destination, set(), incoming)
            for initializer in incoming:
                if provenance_roots(initializer.operands[0]) & workspace_names:
                    raise ValueError(
                        "workspace data was copied into a model-local allocation"
                    )
            if len(incoming) == 1:
                detail = origin(destination)
                if not detail.get("readonly_copy"):
                    raise ValueError("unapproved non-readonly copy in model plumbing")
                readonly_copies.append(detail)
            else:
                materialized_copies.append(
                    {
                        "destination_location": str(
                            destination.owner.operation.location
                        ),
                        "writers": len(incoming),
                        "sources": [
                            str(initializer.operands[0]) for initializer in incoming
                        ],
                    }
                )
        return {
            "schema": 1,
            "status": "PASS",
            "scope": "real local bufferized primitive SSA root continuity; no numerical execution",
            "calls": len(actual_calls),
            "ordered_call_symbols": list(targets),
            "dataflow_verified": True,
            "actual_operand_roots": records,
            "workspace_operands_checked": workspace_operands,
            "destinations_checked": destination_count,
            "workspace_copy_operations": 0,
            "readonly_layout_copies": len(readonly_copies),
            "readonly_layout_copy_details": readonly_copies,
            "tensor_materialization_copies": len(materialized_copies),
            "tensor_materialization_copy_details": materialized_copies,
        }
