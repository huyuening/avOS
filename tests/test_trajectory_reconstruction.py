import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'avos'))
from scenario_reconstructor.object_track_reconstruction import (
    ReconstructionConfig, reconstruct_object_state, correct_heading,
    calculate_reference_heading, _savgol_filter,
)


def track(n=101, speed=10., direction=.5):
    t=np.arange(n)*.1
    state=dict(global_center=np.column_stack([100+speed*np.cos(direction)*t,100+speed*np.sin(direction)*t,np.zeros(n)]),
               heading=np.full(n,direction),valid=np.ones(n,bool),size=np.tile([4.,2.,1.5],(n,1)))
    return t,state


class ReconstructionTests(unittest.TestCase):
    def test_rejects_heading_spike_using_motion(self):
        t,s=track();s['heading'][50]=1.5
        r=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open',filter_method='butterworth'),capture_stages=True)
        self.assertTrue(r.heading_outlier_mask[50]);self.assertAlmostEqual(r.stage_states[2]['heading'][50],.5,places=8)
        self.assertAlmostEqual(r.state['heading'][50],.5,places=6)
        self.assertEqual(r.stage_states[1]['heading'][50],1.5)
        self.assertEqual(s['heading'][50],1.5)

    def test_zero_heading_and_zero_height_are_valid(self):
        t,s=track(direction=0.);s['heading'][50]=1.
        r=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open',filter_method='butterworth'))
        self.assertTrue(r.reconstructed);self.assertTrue(r.heading_outlier_mask[50])
        np.testing.assert_allclose(r.state['heading'],0.,atol=1e-10)
        self.assertTrue(r.state['valid'].all())

    def test_missing_heading_uses_valid_mask_and_stays_within_observed_span(self):
        t,s=track();s['valid'][:3]=False;s['valid'][-4:]=False;s['valid'][40:44]=False
        s['heading'][40:44]=0.;s['global_center'][40:44]=0.
        r=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open',filter_method='butterworth'),capture_stages=True)
        self.assertEqual(r.filled_samples,4);self.assertTrue(np.isnan(r.stage_states[1]['heading'][40:44]).all())
        self.assertTrue(np.isfinite(r.stage_states[2]['heading'][40:44]).all())
        np.testing.assert_array_equal(r.state['valid'][:3],False)
        np.testing.assert_array_equal(r.state['valid'][-4:],False)

    def test_stationary_fallback_does_not_align_to_position_noise(self):
        t,s=track(speed=.1,direction=1.2);s['heading'][:]=-.8
        r=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open',filter_method='butterworth'))
        self.assertFalse(r.heading_outlier_mask.any());np.testing.assert_allclose(r.state['heading'],-.8,atol=1e-8)

    def test_angle_wrap_is_not_rejected(self):
        t=np.arange(11,dtype=float);h=np.linspace(3.,3.3,11);wrapped=np.arctan2(np.sin(h),np.cos(h))
        corrected,rejected=correct_heading(wrapped,h,t)
        self.assertFalse(rejected.any());np.testing.assert_allclose(corrected,h,atol=1e-14)

    def test_reference_preserves_turn_larger_than_half_turn(self):
        t=np.linspace(0,5,101);angle=np.linspace(0,1.8*np.pi,101)
        xy=np.column_stack([10*np.sin(angle),-10*np.cos(angle)])
        ref,moving=calculate_reference_heading(t,xy,angle)
        self.assertTrue(moving.all());self.assertGreater(ref[-1]-ref[0],np.pi)
        self.assertLess(np.max(np.abs(np.diff(ref))),.1)

    def test_sparse_and_absent_anchors_stay_finite(self):
        t=np.arange(5,dtype=float);ref=np.full(5,.5)
        for h in [np.full(5,1.5),np.array([1.5,1.5,.5,1.5,1.5]),np.array([.5,1.5,1.5,1.5,.5])]:
            corrected,rejected=correct_heading(h,ref,t)
            np.testing.assert_allclose(corrected,.5,atol=1e-12)

    def test_short_tracks_and_invalid_timestamps(self):
        t,s=track(2);r=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open',filter_method='butterworth'),capture_stages=True)
        self.assertTrue(r.reconstructed);self.assertTrue(np.isfinite(r.state['heading']).all())
        for bad in [np.array([0.,0.]),np.array([1.,0.]),np.array([0.,np.nan])]:
            self.assertEqual(reconstruct_object_state(s,bad,ReconstructionConfig('waymo-open',filter_method='butterworth')).skipped_reason,'invalid_timestamp')
        s['valid'][1]=False
        self.assertFalse(reconstruct_object_state(s,t,ReconstructionConfig('waymo-open',min_valid_frames=1)).reconstructed)

    def test_capture_does_not_change_result_or_alias_input(self):
        t,s=track();s['heading'][50]=1.5;saved={k:v.copy() for k,v in s.items()}
        plain=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open',filter_method='butterworth'))
        captured=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open',filter_method='butterworth'),capture_stages=True)
        for k in plain.state:np.testing.assert_array_equal(plain.state[k],captured.state[k])
        for k in s:np.testing.assert_array_equal(s[k],saved[k])
        captured.stage_states[1]['global_center'][0,0]=-999
        self.assertNotEqual(s['global_center'][0,0],-999);self.assertNotEqual(captured.state['global_center'][0,0],-999)

class SavitzkyGolayTests(unittest.TestCase):
    def test_quadratic_preserved_and_short_window_supported(self):
        for n in [2, 7, 30, 31, 80]:
            x=np.arange(n,dtype=float);y=.01*x*x+.2*x+1
            np.testing.assert_allclose(_savgol_filter(y),y,atol=1e-10)

    def test_sg_reintegration_preserves_filtered_position_endpoints(self):
        t,s=track();s['global_center'][:,1]+=.05*np.sin(2*t)
        r=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open'),capture_stages=True)
        for axis in [0,1]:
            expected=_savgol_filter(s['global_center'][:,axis])[[0,-1]]
            np.testing.assert_allclose(r.state['global_center'][[0,-1],axis],expected,atol=1e-10)
        self.assertEqual(r.diagnostics['sg_effective_window'],31)
        self.assertEqual(r.diagnostics['reference_low_speed_fraction'],0.)

    def test_sg_low_speed_is_reported_not_hidden(self):
        t,s=track(speed=.1)
        r=reconstruct_object_state(s,t,ReconstructionConfig('waymo-open'))
        self.assertEqual(r.diagnostics['reference_low_speed_fraction'],1.)
        self.assertTrue(r.reconstructed)

    def test_original_sg_minimum_observations(self):
        t,s=track(39)
        self.assertFalse(reconstruct_object_state(s,t,ReconstructionConfig('waymo-open')).reconstructed)

    def test_invalid_sg_configuration_rejected(self):
        for kwargs in [dict(sg_window_length=30),dict(sg_polyorder=31),dict(filter_method='unknown')]:
            with self.assertRaises(ValueError):ReconstructionConfig('waymo-open',**kwargs)

class MovingSelectionTests(unittest.TestCase):
    def test_original_speed_gates_by_type_and_short_observations(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from select_waymo_moving_tracks import motion_eligibility
        t,s=track(speed=.6)
        self.assertTrue(motion_eligibility(s,t,'PEDESTRIAN')['sg_eligible'])
        self.assertFalse(motion_eligibility(s,t,'VEHICLE')['sg_eligible'])
        t,s=track(speed=6.)
        self.assertTrue(motion_eligibility(s,t,'CYCLIST')['sg_eligible'])
        s['valid'][39:]=False
        self.assertFalse(motion_eligibility(s,t,'CYCLIST')['sg_eligible'])

if __name__=='__main__':unittest.main()
