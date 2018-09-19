import torch


def jac_tran_vec(outputs, inputs, vecs):
    dydxs = torch.autograd.grad(outputs, inputs, grad_outputs=vecs,
                                create_graph=True, allow_unused=True)
    return [torch.zeros_like(input_) if dydx is None else dydx
            for (input_, dydx) in zip(inputs, dydxs)]


# https://j-towns.github.io/2017/06/12/A-new-trick.html
def jac_vec(outputs, inputs, vecs):
    jtvs = jac_tran_vec(outputs, inputs, vecs)
    return jac_tran_vec(jtvs, vecs, vecs)


def get_grads(losses, params, **kwargs):
    return [torch.autograd.grad(loss, param, **kwargs)[0]
            for loss, param in zip(losses, params)]


def get_sym_adjs(losses, params):
    grads = get_grads(losses, params, create_graph=True, allow_unused=True)
    hessian_grad_prods = jac_vec(grads, params, grads)
    hessianT_grad_prods = jac_tran_vec(grads, params, grads)
    sym_adjs = [(Ht - H) / 2 for (H, Ht) in
                zip(hessian_grad_prods, hessianT_grad_prods)]
    return sym_adjs


def dot(tensor_list_1, tensor_list_2):
    result = 0
    for tensor_1, tensor_2 in zip(tensor_list_1, tensor_list_2):
        result = result + torch.sum(tensor_1 * tensor_2)
    return result


def sga(losses, params, align=True, epsilon=0.1):
    grads = get_grads(losses, params, create_graph=True, allow_unused=True)
    num_params = sum([param.nelement() for param in params])
    sym_adjs = get_sym_adjs(losses, params)
    if align:
        grad_norm_squared = torch.sum(torch.cat(
            [torch.sum(grad**2).unsqueeze(0) for grad in grads]))
        hamiltonian_grads = [torch.autograd.grad(0.5 * grad_norm_squared,
                                                 param, retain_graph=True)[0]
                             for param in params]
        # grads_cat = torch.cat(grads)
        # hamiltonian_grads_cat = torch.cat(hamiltonian_grads)
        # sym_adjs_cat = torch.cat(sym_adjs)
        lambda_ = torch.sign(dot(grads, hamiltonian_grads) *
                             dot(sym_adjs, hamiltonian_grads) /
                             num_params + epsilon)
    else:
        lambda_ = 1
    return [grad + lambda_ * sym_adj
            for grad, sym_adj in zip(grads, sym_adjs)]


def optimizer_step(optimizer, params, grads):
    for param, grad in zip(params, grads):
        param.grad = grad
    optimizer.step()
