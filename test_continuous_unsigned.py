import unittest
import torch
from continuous_unsigned import Head, get_true_keys, sample, conditional_correlation, aux_labels

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.q, self.v, self.y, _ = sample(128,get_true_keys(1),torch.Generator().manual_seed(12))
    def test_unsigned_scores_reversed(self):
        a=conditional_correlation(self.q,self.v,self.y)
        b=conditional_correlation(self.q,self.v,-self.y)
        torch.testing.assert_close(a,b)
    def test_instant_is_label_blind(self):
        pred=torch.ones_like(self.y)*.31
        label1=aux_labels('instant',self.q,self.v,self.y,pred,torch.Generator().manual_seed(7))
        label2=aux_labels('instant',self.q,self.v,self.y[torch.randperm(len(self.y))],pred,torch.Generator().manual_seed(8))
        torch.testing.assert_close(label1,label2)
        torch.testing.assert_close(label1,self.v.abs().argmax(1))
    def test_model_backward(self):
        model=Head()
        prediction,logits=model(self.q,self.v)
        targets=aux_labels('corr_target',self.q,self.v,self.y,prediction,torch.Generator())
        (torch.nn.functional.mse_loss(prediction,self.y)+.25*torch.nn.functional.cross_entropy(logits,targets)).backward()
        self.assertIsNotNone(model.keys.grad)
        self.assertTrue(torch.isfinite(model.keys.grad).all())

if __name__=='__main__': unittest.main()

