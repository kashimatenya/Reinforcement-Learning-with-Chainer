#モデルを徐々に更新する
class SoftModelUpdater:
    def __init__(self, tau):
        self._tau = tau
    #def

    #モデルを徐々に更新する
    def __call__(self, dst, src):
        for dst_layer, src_layer in zip(dst._hidden_layers, src._hidden_layers):
            self._update_layer(dst_layer, src_layer)
        #for
        
        self._update_layer(dst._output_layer, src._output_layer)
        return
    #def

    #layerごとに更新する
    def _update_layer(self, dst, src):
        tau = self._tau

        dst.W.array *= (1-tau)
        dst.W.array += tau*src.W.array

        if src.b is not None:
            if dst.b is not None:
                dst.b.array *= (1-tau)
                dst.b.array += tau*(src.b.array)
            else:
                dst.b.array = tau*src.b.array
            #if-else
        #if
        return
    #def
#def
