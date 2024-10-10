We add attention coefficient to each node in the neighbourhood.
 The message received by target node will be
 
    h_{v}^{(l)}=\sigma(\sum_{u\in N(v)} \alpha_{vu}W^{(l)}h_{u}^{(l-1)})

where

    \alpha_{vu}= \frac{exp(e_{vu})}{\sum_{k\in N(v)} exp(e_{vk})}

where 

    e_vu=a(W^{(l)}h_u^{l-1}\parallel W^{(l)}h_v^{l-1})

where
a is a feed forward network with a single hidden layer with trainable parameters.
Generally, we train multi-head attention scores($\alpha_{vu}$) and combine them at the end.


    h_{v}^{(l)}[1]=\sigma(\sum_{u\in N(v)} \alpha_{vu}^{[1]}W^{(l)}h_{u}^{(l-1)}) \\
    h_{v}^{(l)}[2]=\sigma(\sum_{u\in N(v)} \alpha_{vu}^{[2]}W^{(l)}h_{u}^{(l-1)})\\
    h_{v}^{(l)}[3]=\sigma(\sum_{u\in N(v)} \alpha_{vu}^{[3]}W^{(l)}h_{u}^{(l-1)})

We aggregate the messages either by concatenating, summation or mean to get the final message/embedding.

    h_{v}^{(l)}=AGG(h_{v}^{(l)}[1],h_{v}^{(l)}[2],h_{v}^{(l)}[3])

This network allows us to dynamically allot different importance to neighbouring nodes in the graph in a relatively computationally efficient manner.
This model Contains Two Head Functions.
