"""Run with python3 -m unittest -v test_protocol.py."""
import unittest
from copy import deepcopy
import numpy as np
from sot_model import SOTSwitchingModel
from coupled_model import NetworkSOTModel

class ProtocolTests(unittest.TestCase):
    def test_allowed_and_rejected_without_mutation(self):
        m=SOTSwitchingModel()
        np.testing.assert_allclose(m.run([0,22,22,0,-22])[:,1], [.3,.41663576827997906,.41663576827997906,.41663576827997906,.3])
        before=(m.m,m.x,deepcopy(m.state))
        with self.assertRaisesRegex(ValueError,'second'): m.step(0)
        self.assertEqual(before,(m.m,m.x,m.state))
        m.reset()
        m.run([0,22,-22])
    def test_fixed_positive_target_after_negative(self):
        m=SOTSwitchingModel()
        m.run([0,-22,0,22])
        self.assertEqual(m.state['pos_target'],29)
        self.assertAlmostEqual(m.m,.41663576827997906)
    def test_partial_resweep(self):
        m=SOTSwitchingModel(); m.run([0,25,20])
        with self.assertRaises(ValueError): m.step(22)
    def test_counts_across_run_calls(self):
        m=SOTSwitchingModel(); m.run([0,22]); m.run([10,-22])
        with self.assertRaises(ValueError): m.run([-18])
    def test_nonfinite_unchanged(self):
        m=SOTSwitchingModel(); old=deepcopy(m.state)
        for x in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError): m.step(x)
            self.assertEqual(m.state,old)
    def test_voltage_wrapper(self):
        m=NetworkSOTModel()
        for va in [0,12,8]: m.step_voltage(8,va,0)
        with self.assertRaises(ValueError): m.step_voltage(8,10,0)

if __name__=='__main__': unittest.main()
