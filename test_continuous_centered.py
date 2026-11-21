import unittest
import torch
from continuous_centered import CenteredHead, sample, teacher
from continuous_unsigned import get_true_keys, conditional_correlation

class CenteredControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def setUp(self):
        self.q,self.x,self.y,self.true=sample(192,get_true_keys(1),torch.Generator().manual_seed(12),1)
    def test_teacher_target_sign_invariant(self):
        pred=torch.zeros_like(self.y)
        r1=teacher('target',self.q,self.x,self.y,pred,torch.Generator().manual_seed(0))
        r2=teacher('target',self.q,self.x,-self.y,pred,torch.Generator().manual_seed(0))
        torch.testing.assert_close(r1,r2)
    def test_instant_teacher_ignores_target(self):
        pred=torch.zeros_like(self.y)
        r1=teacher('instant',self.q,self.x,self.y,pred,torch.Generator().manual_seed(0))
        r2=teacher('instant',self.q,self.x,-self.y,pred,torch.Generator().manual_seed(0))
        torch.testing.assert_close(r1,r2)
        torch.testing.assert_close(r1,(self.x-self.x.mean(1,keepdim=True)).abs().argmax(1))
    def test_forward_backward(self):
        model=CenteredHead()
        p,l=model(self.q,self.x)
        t=teacher('residual',self.q,self.x,self.y,p,torch.Generator().manual_seed(0))
        loss=torch.nn.functional.mse_loss(p,self.y)+.25*torch.nn.functional.cross_entropy(l,t)
        loss.backward()
        self.assertTrue(torch.isfinite(model.keys.grad).all())
        self.assertEqual(sum(x.numel() for x in model.parameters()),19)
    def test_conditional_score_sign_symmetry(self):
        d=self.x-self.x.mean(1,keepdim=True)
        torch.testing.assert_close(conditional_correlation(self.q,d,self.y),
                                   conditional_correlation(self.q,d,-self.y))

if __name__=='__main__':unittest.main()

