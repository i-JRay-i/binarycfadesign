from torch.nn import Module
import cfa_models
import demosaicer_models

class JointModel(Module):
    def __init__(self, P, C):
        super(JointModel, self).__init__()
        self.cfa = cfa_models.LinearFilter(P, C)
        self.demosaicer = demosaicer_models.Demos()
        
    def forward(self, input_tensor):
        raw_image = self.cfa(input_tensor)
        reconstr = self.demosaicer(raw_image)
        return reconstr

class JointModelNoBias(Module):
    def __init__(self, P, C):
        super(JointModelNoBias, self).__init__()
        self.cfa = cfa_models.LinearFilterNoBias(P, C)
        self.demosaicer = demosaicer_models.Demos3D()
        
    def forward(self, input_tensor):
        raw_image = self.cfa(input_tensor)
        [out, x3, x2, x1, psd] = self.demosaicer(raw_image)
        return [out, x3, x2, x1, psd]
    
class JointModelISTANet(Module):
    def __init__(self, P, C):
        super(JointModelNoBias, self).__init__()
        self.cfa = cfa_models.LinearFilterNoBias(P, C)
        self.demosaicer = demosaicer_models.Demos3D()
        
    def forward(self, input_tensor):
        raw_image = self.cfa(input_tensor)
        [out, x3, x2, x1, psd] = self.demosaicer(raw_image)
        return [out, x3, x2, x1, psd]
        
class HenzModel(Module):
    def __init__(self, P, C):
        super(HenzModel, self).__init__()
        self.cfa = cfa_models.LinearFilter(P, C)
        self.demosaicer = demosaicer_models.DemosHenz()
        
    def forward(self, input_tensor, intrpl_filter):
        [filtered_image, weighted_image] = self.cfa(input_tensor)
        reconstr = self.demosaicer(filtered_image, weighted_image, intrpl_filter)
        return reconstr
