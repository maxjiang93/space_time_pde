import sympy
import torch
from torch.autograd import grad
from sympy.parsing.sympy_parser import parse_expr


# utility functions for parsing equations
torch_diff = lambda y, x: grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True,
                               allow_unused=True)[0]


class PDELayer(object):
    """PDE Layer for querying values and computing PDE residues."""

    def __init__(self, in_vars, out_vars):
        """Initialize physics layer.

        Args:
          in_vars: str, a string of input variable names separated by space.
          E.g., 'x y t' for the three variables x, y and t.
          out_vars: str, a string of output variable names separated by space.
          E.g., 'u v p' for the three variables u, v and p.
        """
        self.in_vars = sympy.symbols(in_vars)
        self.out_vars = sympy.symbols(out_vars)
        if not isinstance(self.in_vars, tuple): self.in_vars = (self.in_vars,)
        if not isinstance(self.out_vars, tuple): self.out_vars = (self.out_vars,)
        self.n_in = len(self.in_vars)
        self.n_out = len(self.out_vars)
        self.all_vars = list(self.in_vars) + list(self.out_vars)
        self.eqns_raw = {}  # raw string equations
        self.eqns_fn = {}  # lambda function for the equations
        self.forward_method = None


    def add_equation(self, eqn_str, eqn_name=''):
        """Add an equation to the physics layer.

        The equation string should represent the expression for computing the residue of a given
        equation, rather than representing the equation itself. Use dif(y,x) for computing the
        derivate of y with respect to x. Sign of the expression does not matter. The variable names
        **MUST** be the same as the variables in self.in_vars and self.out_vars.

        E.g.,
        For the equation partial(u, x) + partial(v, y) = 3*partial(u, y)*partial(v, x), write as:
        eqn_str = 'dif(u,x)+dif(v,y)-3*dif(u,y)*dif(v,x)'
        - or -
        eqn_str = '3*dif(u,y)*dif(v,x)-(dif(u,x)+dif(v,y))'

        Args:
          eqn_str: str, a string that can be parsed as an experession for computing the residue of
          an equation.
          eqn_name: str, a name or identifier for this equation entry. E.g., 'div_free'. If none or
          empty, use default of eqn_i where i is an index.

        Raises:
          ValueError: when the variables in the eqn_str do not match that of in_vars and out_vars.

        """
        if not eqn_name:
            eqn_name = 'eqn_{i}'.format(len(self.eqns_raw.keys()))

        # assert that the equation contains the same vars as in_vars and out_vars
        expr = parse_expr(eqn_str)
        valid_var = expr.free_symbols <= (set(self.in_vars)|set(self.out_vars))
        if not valid_var:
            raise ValueError('Variables in the eqn_str ({}) does not match that of '
                             'in_vars ({}) and out_vars ({})'.format(expr.free_symbols,
                                                                     set(self.in_vars),
                                                                     set(self.out_vars)))

        # convert into lambda functions
        fn = sympy.lambdify(self.all_vars, expr, {'dif': torch_diff})

        # update equations
        self.eqns_raw.update({eqn_name: eqn_str})
        self.eqns_fn.update({eqn_name: fn})

    def update_forward_method(self, forward_method):
        """Update forward method.

        Args:
          forward_method: a function, such that y = forward_method(x). x is a tensor of
          shape (..., n_in) and y is a tensor of shape (..., n_out).
        """
        self.forward_method = forward_method

    def eval(self, x):
        """Evaluate the output values using forward_method.

        Args:
          x: a tensor of shape (..., n_in)
        Returns:
          a tensor of shape (..., n_out)
        """
        if not self.forward_method:
            raise RuntimeError('forward_method has not been defined.'
                               'Run update_forward_method first.')
        y = self.forward_method(x)
        if not ((x.shape[-1] == self.n_in) and (y.shape[-1] == self.n_out)):
            raise ValueError('Input/output dimensions ({}/{}) not equal to the dimensions of '
                             'defined variables ({}/{}).'.format(x.shape[-1], y.shape[-1],
                                                                 self.n_in, self.n_out))
        return y

    def __call__(self, x, return_residue=True):
        """Compute the forward eval and possibly compute residues from the previously defined pdes.

        Args:
          x: input tensor of shape (..., n_in)
          return_residue: bool, whether to return the residue of the pde for each equation.
        Returns:
          y: output tensor of shape (..., n_out)
          residues (optional): a dictionary containing residue evaluation for each pde.
        """

        if not return_residue:
            y = self.eval(x)
            return y
        else:
            # split into individual channels and set each to require grad.
            inputs = [x[..., i:i+1] for i in range(x.shape[-1])]
            for xx in inputs:
              xx.requires_grad = True
            x_ = torch.cat(inputs, axis=-1)
            y = self.eval(x_)
            outputs = [y[..., i:i+1] for i in range(y.shape[-1])]
            inputs_outputs = inputs + outputs
            residues = {}
            for key, fn in self.eqns_fn.items():
              residue = fn(*inputs_outputs)
              residues.update({key: residue})
            return y, residues