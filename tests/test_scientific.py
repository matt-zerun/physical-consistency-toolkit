"""Analytical tests of S1-S4; distinct from empirical-result reproduction."""
import unittest
import numpy as np
from physical_consistency import Surface,compare_surfaces,components
from physical_consistency.pci import gradients


class ScientificTests(unittest.TestCase):
    def setUp(self):
        self.w=np.linspace(0,4,9);self.d=np.linspace(1,2,6)
        self.W,self.D=np.meshgrid(self.w,self.d,indexing="ij")
    def surface(self,z,mask=None):
        return Surface(self.w,self.d,z,mask)
    def test_constant(self):
        t=compare_surfaces({"constant":self.surface(np.ones_like(self.W))})
        self.assertEqual(t.pci.iloc[0],0)
        self.assertEqual(t.rough_ref.iloc[0],0)
    def test_affine_coordinate_derivatives(self):
        s=self.surface(0.3+0.1*self.W+0.6*self.D)
        a,b,v=gradients(s)
        np.testing.assert_allclose(a[v],0.1,atol=1e-14)
        np.testing.assert_allclose(b[v],0.6,atol=1e-14)
        c=components(s)
        self.assertAlmostEqual(c["roughness"],np.sqrt(0.1**2+0.6**2))
        self.assertEqual(c["mvr"],0)
        self.assertEqual(c["pbv"],0)
    def test_one_decreasing_axis(self):
        c=components(self.surface(3-0.1*self.W+0.1*self.D))
        self.assertEqual(c["mvr"],0.5)
    def test_two_decreasing_axes(self):
        self.assertEqual(components(self.surface(4-0.1*self.W-0.1*self.D))["mvr"],1)
    def test_configurable_direction(self):
        c=components(self.surface(3-0.1*self.W+0.1*self.D),directions=(-1,1))
        self.assertEqual(c["mvr"],0)
    def test_unconstrained_axis_extension(self):
        c=components(self.surface(3-0.1*self.W+0.1*self.D),directions=(0,1))
        self.assertEqual(c["mvr"],0)
        self.assertGreater(c["roughness"],0)
    def test_bounds_and_endpoints(self):
        z=np.full_like(self.W,0.2);z[0]=5;z[1]=5.01;z[2]=0.19
        c=components(self.surface(z))
        self.assertAlmostEqual(c["pbv"],2/len(self.w))
    def test_joint_maximum_normalization_and_weights(self):
        t=compare_surfaces({"a":self.surface(1+0.1*self.W),"b":self.surface(1+0.2*self.W)})
        np.testing.assert_allclose(t.rough_norm,[0.5,1])
        np.testing.assert_allclose(t.pci,[0.2,0.4])
        u=compare_surfaces({"a":self.surface(1+0.1*self.W),"b":self.surface(1+0.2*self.W)},weights=[0.2,0.7,0.1])
        np.testing.assert_allclose(u.pci,[0.35,0.7])
    def test_invalid_legacy_weight_sum(self):
        with self.assertRaises(ValueError):
            compare_surfaces({"x":self.surface(self.W)},weights=[0.4,0.8,0.2])
    def test_irregular_checkerboard(self):
        z=1+0.2*((np.indices(self.W.shape).sum(axis=0)%2)*2-1)
        c=components(self.surface(z))
        self.assertGreater(c["mvr"],0.3)
        self.assertGreater(c["roughness"],1)
    def test_quadratic_first_order_and_backward_boundary(self):
        # f=w^2 + d^2: forward derivatives 2w+h, backward 2w-h at last node.
        a,b,v=gradients(self.surface(self.W**2+self.D**2))
        ew=2*self.W+(self.w[1]-self.w[0]);ew[-1]=2*self.W[-1]-(self.w[1]-self.w[0])
        ed=2*self.D+(self.d[1]-self.d[0]);ed[:,-1]=2*self.D[:,-1]-(self.d[1]-self.d[0])
        np.testing.assert_allclose(a,ew,atol=1e-12)
        np.testing.assert_allclose(b,ed,atol=1e-12)
    def test_refinement_convergence(self):
        errors=[]
        for n in [11,21,41]:
            w=np.linspace(0,1,n);d=np.linspace(0,1,n)
            W,D=np.meshgrid(w,d,indexing="ij")
            a,b,v=gradients(Surface(w,d,W**2+D**2))
            errors.append(np.max(np.abs(a-2*W)))
        np.testing.assert_allclose(errors,[0.1,0.05,0.025],atol=1e-12)
    def test_mask_does_not_cross_outside_nodes(self):
        mask=self.W+self.D<4
        s=self.surface(1+0.1*self.W+0.2*self.D,mask)
        a,b,v=gradients(s)
        np.testing.assert_allclose(a[v],0.1,atol=1e-14)
        np.testing.assert_allclose(b[v],0.2,atol=1e-14)
        self.assertTrue(np.all(v<=mask))
        self.assertEqual(components(s)["mvr"],0)
    def test_masks_must_match(self):
        with self.assertRaises(ValueError):
            compare_surfaces({"a":self.surface(self.W),"b":self.surface(self.W,self.W<3)})
    def test_nan_inside_domain_fails(self):
        z=self.W.copy();z[1,1]=np.nan
        with self.assertRaises(ValueError): self.surface(z)
    def test_axis_regular_and_shape_validation(self):
        with self.assertRaises(ValueError): Surface([0,1,3],[1,2],np.zeros((3,2)))
        with self.assertRaises(ValueError): Surface([0,1],[1,2],np.zeros((3,2)))
    def test_mvr_tolerance_is_explicit(self):
        s=self.surface(1-1e-9*self.W)
        self.assertEqual(components(s)["mvr"],0.5)
        self.assertEqual(components(s,tolerance=1e-8)["mvr"],0)
    def test_smooth_steep_surface_penalty(self):
        t=compare_surfaces({"gentle":self.surface(1+0.05*self.W),"steep":self.surface(1+0.5*self.W)})
        self.assertGreater(t.pci.iloc[1],t.pci.iloc[0])
    def test_zero_weights_and_zero_directions(self):
        c=components(self.surface(self.W),directions=(0,0))
        self.assertEqual(c["mvr"],0)
        for weights in [[0,0,0],[-0.1,0.9,0.2],[np.nan,0.8,0.2]]:
            with self.assertRaises(ValueError):compare_surfaces({"x":self.surface(self.W)},weights=weights)

if __name__=="__main__":unittest.main()
