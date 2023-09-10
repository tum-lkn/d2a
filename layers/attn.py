"""
Implements attention layers used.
"""
import torch
from sparsemax import Sparsemax
import numpy as np
import networkx as nx
from collections import OrderedDict
from typing import Tuple, Union, List, Dict, Any, Iterator


class ModuleMutationMixin(object):
    def mutate(self, num_samples, dev='cpu'):
        for module in self:
            if hasattr(module, 'mutate'):
                module.mutate(num_samples, dev)

    def set_training(self, value):
        self.training = value
        for module in self:
            module.training = value

    def __iter__(self) -> Iterator[torch.nn.Module]:
        return iter(self._modules.values())


class MultiHeadAttentionModuleConfig(object):
    """
    Represent the configuration of an attention module. An attention module
    consists of a multi-head attention layer followed by a fully connected
    layer and potentially a batch normalization layer.
    """
    @classmethod
    def from_dict(cls, dc: Dict[str, Any]) -> 'MultiHeadAttentionModuleConfig':
        return cls(**dc)

    def __init__(self, num_heads: int, dim_fcn: int, dim_hidden: int, dim_in: int,
                 dim_out: int, fcn_activation: str='relu',
                 dim_q: int=None, dim_v: int=None, dim_k: int=None,
                 attention_class='SelfAttentionLayer', attn_activation: str='softmax'):
        """
        Initializes object.
        Args:
            num_heads: Number of attention heads.
            dim_in: Size of the input for keys, values and queries, can be
                further specified with dim_q, dim_v and dim_k.
            dim_fcn: Size of dense layer following the multi-head attention layer.
            dim_out: Output dimensionality of each Self-Attention layer.
            fcn_activation: Name of the activation function for the hidden layer.
            pool: Pooling function used after applying the non-linear transform.
                Must be in {max, squeeze, sum, mean}.
            batch_norm: Whether a batch-normalization should occur at the end.
            attention_class: Class that is repeated.
            attn_activation: Activation function for the attention weights.
                Possible values are sparsemax and softmax.
        """
        self.num_heads = num_heads
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.dim_fcn = dim_fcn
        self.fcn_activation = fcn_activation
        self.attention_class = attention_class
        self.dim_q = dim_q
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.attn_activation = attn_activation

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class SelfAttentionModuleConfig(object):
    """
    Represent the configuration of an attention module. An attention module
    consists of a multi-head attention layer followed by a fully connected
    layer and potentially a batch normalization layer.
    """
    @classmethod
    def from_dict(cls, dc: Dict[str, Any]) -> 'SelfAttentionModuleConfig':
        return cls(**dc)

    def __init__(self, dim_fcn: int, dim_hidden: int, dim_in: int,
                 dim_out: int, dim_q: int=None, dim_v: int=None, dim_k: int=None,
                 fcn_activation: str='relu', attn_activation: str='softmax'):
        """
        Initializes object.
        Args:
            dim_in: Size of the input for keys, values and queries, can be
                further specified with dim_q, dim_v and dim_k.
            dim_out: Output dimensionality of layer.
            dim_q: Size of latent space queries are projected to.
            dim_v: Size of latent space values are projected to.
            dim_k: Size of latent space keys are projected to.
            dim_fcn: Size of dense layer following the multi-head attention layer.
            fcn_activation: Name of the activation function for the hidden layer.
            attn_activation: Activation function for the attention weights.
                Possible values are sparsemax and softmax.
        """
        self.dim_out = dim_out
        self.dim_q = dim_q
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_fcn = dim_fcn
        self.fcn_activation = fcn_activation
        self.attn_activation = attn_activation

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class SelfAttentionLayer(torch.nn.Module, ModuleMutationMixin):
    """
    Implements a special purpose self attention layer for predicting the
    performance of a single VNF.
    The main difference is that queries has a different feature dimension as
    values and keys for the first layer. Thus, this class provides a special
    parameter that allows the user to manually set the corresponding dimension
    of the transformation matrix for the queries.
    """
    @classmethod
    def from_config(cls, config: SelfAttentionModuleConfig) -> 'SelfAttentionLayer':
        return cls(
            dim_in=config.dim_in,
            dim_q=config.dim_q,
            dim_v=config.dim_v,
            dim_k=config.dim_k,
            dim_hidden=config.dim_hidden,
            dim_out=config.dim_out,
            activation_f=config.attn_activation
        )

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dim_q=None,
                 dim_v=None, dim_k=None, activation_f='softmax', linear_class='Linear'):
        """

        Args:
            dim_in: Dimensionality of input, i.e., dimension of last axis.
            dim_hidden: Dimensionality of hidden embedding space for keys and
                queries.
            dim_out: Dimensionality of hidden embedding space for values.
            dim_q: Dimensionality of last dimension of queries if different
                from argument `dim_in`.
            dim_v: Dimensionality of last dimension of values if different
                from argument `dim_in`.
            dim_k: Dimensionality of last dimension of keys if different
                from argument `dim_in`.
        """
        super(SelfAttentionLayer, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_in = dim_in
        self.dim_q = dim_q
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.dim_out = dim_out
        self.linear_class = linear_class

        linear_cls = LINEAR_FACTORY[linear_class]

        self.transform_keys = linear_cls(
            in_features=dim_in if dim_k is None else dim_k,
            out_features=dim_hidden,
            bias=False
        )
        self.transform_queries = linear_cls(
            in_features=dim_in if dim_q is None else dim_q,
            out_features=dim_hidden,
            bias=False
        )
        self.transform_values = linear_cls(
            in_features=dim_in if dim_v is None else dim_v,
            out_features=dim_out,
            bias=False
        )
        if activation_f == 'softmax':
            self.activation = torch.nn.Softmax(dim=-1)
        elif activation_f == 'sparsemax':
            self.activation = Sparsemax(dim=-1)
        else:
            raise KeyError("Unknown attention atcitvation function {}".format(activation_f))

    def _unmasked_attention(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Swap the last two dimensions of the keys. Queries has shape
        (BS, T_max, dim_hidden), transposed keys (BS, dim_hidden, T_max).
        Result has shape (BS, T_max, T_max).
        """
        scores = torch.matmul(queries, torch.transpose(keys, dim0=-2, dim1=-1))
        scores = self._apply_activation(scores)
        return scores

    @classmethod
    def _make_outer_mask(cls, query_mask: torch.Tensor):
        """
        Transform the query mask to shape (BS, 1, T_max) and assign a
        negative large value to the zero elements. This results in a
        weight of zero after applying the activation function.
        In[3]: query_mask
        Out[3]: tensor([[[0.],
                         [1.],
                         [0.]],

                        [[0.],
                         [1.],
                         [1.]]])
        In[3]: (1 - torch.transpose(query_mask, -2, -1)) * -1e9
        Out[3]: tensor([[[-1e+09,  0, -1e+09],

                        [[-1e+09,  0,  0]]]),
        """
        outer_mask = (1 - torch.transpose(query_mask, -2, -1)) * -1e9
        return outer_mask

    @classmethod
    def _mask_scores(cls, outer_mask: torch.tensor, scores: torch.tensor) -> torch.tensor:
        """
        Add the outer mask to the values. The entries that are zero in the
        query mask and associate with negative large values in the outer
        mask result in a weight close to zero for the corresponding
        scores. The outer_mask is broadcasted along the second dimension
        of each batch.
        In[6]; scores
        Out[6]: tensor([[[ 0.2497,  1.2428, -0.4236],
                         [ 1.2475, -1.4368,  1.1158],
                         [ 0.8214, -1.7751, -1.3290]],
                        [[-0.3541,  0.9931, -0.2170],
                         [ 0.7973, -1.5163,  2.7247],
                         [ 1.6784,  0.6680,  1.0098]]])
        In[7]: torch.add(scores, outer_mask)
        Out[7]: tensor([[[-1.0000e+09,  1.2428e+00, -1.0000e+09],
                         [-1.0000e+09, -1.4368e+00, -1.0000e+09],
                         [-1.0000e+09, -1.7751e+00, -1.0000e+09]],
                        [[-1.0000e+09,  9.9312e-01, -2.1698e-01],
                         [-1.0000e+09, -1.5163e+00,  2.7247e+00],
                         [-1.0000e+09,  6.6800e-01,  1.0098e+00]]])
        """
        scores = torch.add(outer_mask, scores)
        return scores

    def _apply_activation(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply activation function to obtain the weights. The outer_mask
        pulls the weights of those entries that are zero in the mask to
        zero in the activation function. This affects the entries that
        combine zero elements and non-zero elements. For the rows that
        are zero, equal weights are returned.
        In[8]: softmax(scores / sqrt(self.dim_out))
        Out[8]: tensor([[[0.0000, 1.0000, 0.0000],
                         [0.0000, 1.0000, 0.0000],
                         [0.0000, 1.0000, 0.0000]],
                        [[0.0000, 0.7703, 0.2297],
                         [0.0000, 0.0142, 0.9858],
                         [0.0000, 0.4154, 0.5846]]])
        """
        return self.activation(scores / np.sqrt(float(self.dim_out)))

    def _masked_attention(self, attention_mask: torch.Tensor, queries: torch.Tensor,
                          keys: torch.Tensor) -> torch.Tensor:
        """
        Assign an attention weight of zero to those values, that should not be
        attended to according to the attention mask.
        """
        scores = self._apply_activation(self._mask_scores(
            outer_mask=self._make_outer_mask(attention_mask),
            scores=torch.matmul(queries, torch.transpose(keys, dim0=-2, dim1=-1))
        ))
        return scores

    def forward(self, keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor,
                attention_mask=None, attention_weights=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform attention transformation.

        Args:
            keys (tensor): Tensor with three dimensions: (BS, T_max, ft_keys).
            queries (tensor): Tensor with three dimensions: (BS, T_max, ft_queries).
            values (tensor): Tensor with three dimensions: (BS, T_max, ft_values).
            attention_mask(tensor): Binary tensor of shape: (BS, T_max, 1).
            attention_weights(tensor): Tensor of shape (BS, T_max, T_max),
                precalculated attention weights. If used, then
                attention weights are not calculated in this layer but these
                are used instead.

        Note:
            The argument `attention_mask` can be used to handle variable sized
            inputs. Zero values indicate that the corresponding row in values
            should not be attended to. During the attention weight calculation
            a value of zero will thus be assigned to those entries.

        Returns:
            outputs: Tensor of shape (BS, T_max, dim_hidden).
            scores: Tensor of shape (BS, T_max, T_max).
        """
        values = self.transform_values(values)
        if attention_weights is None:
            keys = self.transform_keys(keys)
            queries = self.transform_queries(queries)
            if attention_mask is None:
                scores = self._unmasked_attention(queries, keys)
            else:
                scores = self._masked_attention(attention_mask, queries, keys)
        else:
            scores = attention_weights
        outputs = torch.matmul(scores, values)
        return outputs, scores

    def get_config(self):
        return {
            "dim_hidden": self.dim_hidden,
            "dim_in": self.dim_in,
            "dim_q": self.dim_q,
            "dim_v": self.dim_v,
            "dim_k": self.dim_k,
            "dim_out": self.dim_out
        }


class MultiHeadAttentionLayer(torch.nn.Module, ModuleMutationMixin):
    """
    Stacks multiple SelfAttentionLayers in parallel.
    """

    @classmethod
    def from_config(cls, config: MultiHeadAttentionModuleConfig):
        return cls(
            num_heads=config.num_heads,
            attention_class=config.attention_class,
            dim_in=config.dim_in,
            dim_hidden=config.dim_hidden,
            dim_out=config.dim_out,
            dim_q=config.dim_q,
            dim_v=config.dim_v,
            dim_k=config.dim_k,
            activation_f=config.attn_activation
        )

    def __init__(self, num_heads: int, attention_class: Union[str, callable],
                 dim_in: int, dim_hidden: int, dim_out: int, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()
        def listify(x):
            if type(x) in [int, np.int32, np.int64, np.int]:
                return [x] * num_heads
            else:
                assert len(x) == num_heads, 'length of dim_hidden or dim_out not equal to the number of heads'
                return x

        factory = {'SelfAttentionLayer': SelfAttentionLayer, 'Gat': Gat}
        if 'kwargs' in kwargs:
            # this happens when restoring directly from a stored config file.
            # calling MultiHeadAttentionLayer(**restored_config_json) then adds
            # the kwargs entry in kwargs argument.
            kwargs = kwargs['kwargs']
        self.kwargs = kwargs
        self.num_heads = num_heads
        self.dim_hidden = listify(dim_hidden)
        self.dim_out = listify(dim_out)
        self.dim_in = listify(dim_in)
        self.attention_class = attention_class

        if type(attention_class) == str:
            attention_class = factory[attention_class]

        layers = [attention_class(
            dim_hidden=h,
            dim_out=o,
            dim_in=i,
            **kwargs
        ) for h, o, i in zip(self.dim_hidden, self.dim_out, self.dim_in)]
        # Wrap layers in ModuleList in order to make the parameters of those
        # layers available in the .parameter() method of this layer.
        self.layers = torch.nn.ModuleList(layers)

    def mutate(self, num_samples, dev='cpu'):
        for module in self.layers:
            if hasattr(module, 'mutate'):
                module.mutate(num_samples, dev)

    def set_training(self, value):
        self.training = value
        for module in self.layers:
            module.training = value

    def forward(self, **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Execute multiple heads in parallel.

        Args:
            keys (tensor): Tensor with three dimensions: (BS, T_max, ft_keys).
            queries (tensor): Tensor with three dimensions: (BS, T_max, ft_queries).
            values (tensor): Tensor with three dimensions: (BS, T_max, ft_values).
            attention_mask (tensor): Binary tensor of shape: (BS, T_max, 1).
            attention_weights(tensor): Tensor of shape (BS, num_heads, T_max, T_max),
                precalcualted attention weights for every head.

        Returns:
            outputs: Tensor of shape (BS, T_max, dim_hidden * num_heads).
            scores: List of Tensor of shape (BS, T_max, T_max). List has length num_heads.
        """
        scores = []
        returns = []
        for i, l in enumerate(self.layers):
            r, s = l(**kwargs)
            scores.append(s)
            returns.append(r)
        return torch.cat(returns, axis=-1), scores

    def get_config(self):
        return {
            "num_heads": self.num_heads,
            "attention_class": self.attention_class,
            "dim_hidden": self.dim_hidden,
            "dim_out": self.dim_out,
            "dim_in": self.dim_in,
            "kwargs": self.kwargs
        }


class AttentionModule(torch.nn.Module):
    """
    Module that combines an arbitrary attention layer with a linear layer.
    """
    def __init__(self, config: Union[MultiHeadAttentionModuleConfig, SelfAttentionModuleConfig]):
        super(AttentionModule, self).__init__()
        factory_cls = {
            MultiHeadAttentionModuleConfig: MultiHeadAttentionLayer,
            SelfAttentionModuleConfig: SelfAttentionLayer
        }
        calc_attn_out_dim = {
            MultiHeadAttentionModuleConfig: lambda x: x.num_heads * x.dim_out,
            SelfAttentionModuleConfig: lambda x: x.dim_out
        }
        self.attention_layer = factory_cls[type(config)].from_config(config)
        self.linear = torch.nn.Linear(
            in_features=calc_attn_out_dim[type(config)](config),
            bias=True,
            out_features=config.dim_fcn
        )
        self.activation = torch.nn.ReLU()
        self.last_scores = None

    def forward(self, keys: torch.Tensor, values: torch.Tensor, queries: torch.Tensor,
                attention_mask=None, weights=None) -> torch.Tensor:
        """
        Execute multiple heads in parallel.

        Args:
            keys (tensor): Tensor with three dimensions: (BS, T_max, ft_keys).
            queries (tensor): Tensor with three dimensions: (BS, T_max, ft_queries).
            values (tensor): Tensor with three dimensions: (BS, T_max, ft_values).
            attention_mask (tensor): Binary tensor of shape: (BS, T_max, 1).

        Returns:
            outputs: Tensor of shape (BS, .., T_max, out_dim).
        """
        ret, scores = self.attention_layer(
            keys=keys,
            values=values,
            queries=queries,
            attention_mask=attention_mask,
            attention_weights=weights
        )
        self.last_scores = scores
        return self.activation(self.linear(ret))


class SequentialAttention(torch.nn.Module):
    """
    An adaptation of the pytorch sequential container to the special needs
    of attention modules.
    """
    def __init__(self, args: Union[List[AttentionModule], OrderedDict]):
        super(SequentialAttention, self).__init__()
        if type(args) == OrderedDict:
            for k, v in args.items():
                self.add_module(k, v)
        else:
            for i, m in enumerate(args):
                self.add_module('module-{:d}'.format(i), m)

    def __iter__(self) -> Iterator[AttentionModule]:
        return iter(self._modules.values())

    def forward(self, keys: torch.Tensor, values: torch.Tensor, queries: torch.Tensor,
                attention_mask=None, reuse_queries=False) -> torch.Tensor:
        """
        Execute the attention modules stored in this class and returns the output
        of the last one.

        For each next module, the return values of the preceeding module are
        used as input for keys and values. For the queries, those can be reused
        if reuse_queries is set to true. If reuse_queries=False, then the return
        values of the previous module are also used as queries.

        Args:
            keys (tensor): Tensor with three dimensions: (BS, T_max, ft_keys).
            queries (tensor): Tensor with three dimensions: (BS, T_max, ft_queries).
            values (tensor): Tensor with three dimensions: (BS, T_max, ft_values).
            attention_mask (tensor): Binary tensor of shape: (BS, T_max, 1).
            reuse_queries (bool): Whether or not to keep the queries the same
                across levels.

        Returns:
            outputs: Tensor of shape (BS, .., T_max, out_dim).
        """
        ret = None
        for module in self:
            ret = module(keys=keys, values=values, queries=queries, attention_mask=attention_mask)
            keys = ret
            values = ret
            queries = queries if reuse_queries else ret
        return ret


class Gat(torch.nn.Module, ModuleMutationMixin):

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dim_q=None,
                 dim_v=None, dim_k=None, activation_f='softmax', linear_class='Linear'):
        super(Gat, self).__init__()
        self.attention_module = SelfAttentionLayer(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            dim_q=dim_q,
            dim_v=dim_v,
            dim_k=dim_k,
            activation_f=activation_f,
            linear_class=linear_class
        )

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
        """
        Compute new features for each node. New features are calculated based
        on the convex combination of the neighborhood of the node which is
        indicated in the adj argument.

        Args:
            node_features: Tensor with node fatures, has shape (num_nodes, num_node_ft).
            adj: The adjacency information, has shape (num_nodes, max_neighborhood_size, 1),
                where max_neighborhood_size can be, e.g., the maximum node degree + 1
                for a one-hop neighborhood. The dtype must be torch.long.
            mask: Masks out entries in adj for nodes that do not have the full
                degree. Has shape (num_nodes, max_neighborhood_size, 1) and a
                dtype of torch.float.

        Returns:
            out: The new node features of shape (num_nodes, dim_out).
            weights: The individual attention weights.
        """
        tmp = torch.reshape(adj, [adj.shape[0], adj.shape[1] * adj.shape[2], adj.shape[3]])
        fts = torch.gather(node_features, 1, tmp.repeat_interleave(node_features.shape[-1], -1))
        fts = torch.reshape(
            fts,
            [adj.shape[0], adj.shape[1], adj.shape[2], node_features.shape[-1]]
        )

        # kv = torch.reshape(
        #     node_features[adj.flatten(), :],
        #     [adj.shape[0], adj.shape[1], node_features.shape[1]]
        # )
        out, weights = self.attention_module(
            keys=fts,
            values=fts,
            queries=torch.unsqueeze(node_features, dim=-2),
            attention_mask=mask
        )
        return torch.squeeze(out, dim=-2), weights


class MutateLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super(MutateLinear, self).__init__()
        self.dim_in = in_features
        self.dim_out = out_features
        self.use_bias = bias
        self.noise_scale = torch.tensor(1, dtype=torch.float32)

        self.weight = torch.nn.Parameter(
            torch.empty([self.dim_in, self.dim_out], dtype=torch.float32),
            requires_grad=False
        )
        torch.nn.init.xavier_normal(self.weight)
        if self.use_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros([1, self.dim_out], dtype=torch.float32),
                requires_grad=False
            )

        self.mutated_weight = None
        self.mutated_bias = None

        self.noise_weight = None
        self.noise_bias = None

        self.best_weight = None
        self.best_bias = None

    def mutate(self, num_samples, dev='cpu'):
        self.noise_weight = self.noise_scale * torch.randn(
            [num_samples, self.dim_in, self.dim_out],
            dtype=torch.float32,
            requires_grad=False,
            device=dev
        )
        self.mutated_weight = torch.unsqueeze(self.weight, 0) + self.noise_weight
        if self.use_bias:
            self.noise_bias = torch.randn(
                [num_samples, 1, self.dim_out],
                dtype=torch.float32,
                requires_grad=False,
                device=dev
            )
            self.mutated_bias = torch.unsqueeze(self.bias, 0) + self.noise_bias

    def decrease_scale(self):
        self.noise_scale = torch.clamp_min(self.noise_scale * 0.9999, 1e-4)

    def forward(self, x: torch.Tensor):
        if self.training:
            w = self.mutated_weight
            tmp = torch.matmul(x, w)
            # print(tmp.shape)
            if self.use_bias:
                w = self.mutated_bias
                tmp = torch.add(tmp, w)
                # print(tmp.shape)
        else:
            tmp = torch.matmul(x, self.weight.unsqueeze(0))
            # tmp = torch.matmul(x, self.best_weight.unsqueeze(0))
            # print(tmp.shape)
            if self.use_bias:
                tmp = torch.add(tmp, self.bias.unsqueeze(0))
                # tmp = torch.add(tmp, self.best_bias.unsqueeze(0))
                # print(tmp.shape)
        return tmp


FACTORY = {
    'SelfAttentionLayer': SelfAttentionLayer,
    'MultiHeadAttentionLayer': MultiHeadAttentionLayer,
    'Gat': Gat
}


LINEAR_FACTORY = {
    'Linear': torch.nn.Linear,
    'MutateLinear': MutateLinear
}


if __name__ == '__main__':
    num_nodes = 23
    graphs = [nx.erdos_renyi_graph(num_nodes, 0.5) for i in range(5)]
    max_degree = np.max([np.max(list(dict(g.degree()).values())) for g in graphs])

    link_lists = []
    masks = []
    for g in graphs:
        adj = np.zeros([1, num_nodes, max_degree, 1], dtype=np.int)
        msk = np.zeros([1, num_nodes, max_degree, 1], dtype=np.float32)
        indices = {n: 0 for n in g.nodes()}
        for u, v in g.edges():
            idx = indices[u]
            adj[0, u, idx, 0] = v
            msk[0, u, idx, 0] = 1
            indices[u] += 1
            idx = indices[v]
            adj[0, v, idx, 0] = u
            msk[0, v, idx, 0] = 1
            indices[v] += 1
        link_lists.append(adj)
        masks.append(msk)

    adjs = np.concatenate(link_lists)
    masks = np.concatenate(masks)
    adjs_t = torch.tensor(adjs)
    masks_t = torch.tensor(masks)
    node_fts = np.reshape(np.arange(num_nodes), [1, 23, 1]).astype(np.float32)
    node_fts = np.repeat(np.repeat(node_fts, axis=2, repeats=7), axis=0, repeats=5)
    node_fts_t = torch.tensor(node_fts)

    tmp = torch.reshape(adjs_t, [adjs_t.shape[0], adjs_t.shape[1] * adjs_t.shape[2], adjs_t.shape[3]])
    fts = torch.gather(node_fts_t, 1, tmp.repeat_interleave(node_fts_t.shape[-1], -1))
    fts = torch.reshape(
        fts,
        [adjs_t.shape[0], adjs_t.shape[1], adjs_t.shape[2], node_fts_t.shape[-1]]
    )
