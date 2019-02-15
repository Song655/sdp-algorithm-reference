"""Unit tests for pipelines


"""

import os
import unittest

import sys
sys.path.append(os.path.join('..', '.'))

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data.polarisation import PolarisationFrame
from arl.image.operations import export_image_to_fits, create_empty_image_like
from arl.pipelines.functions import continuum_imaging, spectral_line_imaging, ical, rcal, eor, fast_imaging
from arl.skycomponent.operations import create_skycomponent
from arl.util.testing_support import create_named_configuration, create_test_image, create_blockvisibility_iterator
from arl.visibility.operations import append_visibility
from arl.visibility.base import create_blockvisibility
from arl.imaging import predict_2d

import logging
import time

log = logging.getLogger(__name__)


fmt = '%(asctime)s %(filename)s[line: %(lineno)d] %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=fmt,
                    filename='ocl_logs.txt',
                    filemode='w',
                    datefmt='%a, %d %b %Y %H:%M:%S'
                    )


class TestPipelines(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        lowcore = create_named_configuration('LOWBD2-CORE')
        times =  numpy.linspace(-100, 100, 100)* numpy.pi / 12.0
        vnchan = 8
        frequency = numpy.linspace(0.8e8, 1.20e8, vnchan)
        channel_bandwidth = numpy.array(vnchan * [frequency[1] - frequency[0]])
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.9 * f, 0.8 * f, 0.7 * f, 0.6 * f, 0.5 * f, 0.4 * f, 0.3 * f])
        #self.flux = numpy.array([f, 0.9 * f, 0.8 * f])

        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
        
        self.comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compabsdirection)
        self.image = create_test_image(canonical = True, frequency=frequency, phasecentre=self.phasecentre, cellsize=0.001,
                                       polarisation_frame=PolarisationFrame('stokesI'))
        
        self.blockvis = create_blockvisibility_iterator(lowcore, times=times, frequency=frequency,
                                                        channel_bandwidth=channel_bandwidth,
                                                        phasecentre=self.phasecentre, weight=1,
                                                        polarisation_frame=PolarisationFrame('stokesI'),
                                                        integration_time=1.0, number_integrations=1, predict=predict_2d,
                                                        components=self.comp, phase_error=0.1, amplitude_error=0.01,
                                                        sleep=1.0)
        
        self.vis = create_blockvisibility(lowcore, times=times, frequency=frequency,
                                     channel_bandwidth=channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1,
                                     polarisation_frame=PolarisationFrame('stokesI'),
                                     integration_time=1.0)
        
        self.vis = predict_2d(self.vis, self.image, padding=1)
    
    def ingest(self):
        vis = None
        for iv, subvis in enumerate(self.blockvis):
            if iv == 0:
                vis = subvis
            else:
                vis = append_visibility(vis, subvis)
        
        return vis
    """
    def test_RCAL(self):
        for igt, gt in enumerate(rcal(vis=self.blockvis, components=self.comp)):
            log.info("Chunk %d, gaintable size %.3f (GB)" % (igt, gt.size()))
    
    def test_ICAL(self):
        icalpipe = ical(vis=self.vis, components=self.comp)
    """
    def test_continuum_imaging(self):
        model = create_empty_image_like(self.image)
        start = time.time() 
        visres, comp, residual = continuum_imaging(self.vis, model, algorithm='msmfsclean',
                                                   scales=[0, 3, 10, 20], threshold=0.01, nmoments= 2, findpeak='ARL',
                                                   fractional_threshold=0.01, psf_support = 200,nmajor=1, padding=1, niter=200)
        elapsed = (time.time() - start)
        print("Time used for test_continuum_imaging: %f s" % elapsed)
        export_image_to_fits(comp, "%s/test_pipelines-continuum-imaging-comp.fits" % (self.dir))
    """
    def test_continuum_imaging_psf(self):
        model = create_empty_image_like(self.image)
        visres, comp, residual = continuum_imaging(self.vis, model, algorithm='msmfsclean', psf_width=100,
                                                   scales=[0, 3, 10], threshold=0.01, nmoments=2, findpeak='ARL',
                                                   fractional_threshold=0.01)
        export_image_to_fits(comp, "%s/test_pipelines-continuum-imaging_psf-comp.fits" % (self.dir))

    def test_spectral_line_imaging_no_deconvolution(self):
        model = create_empty_image_like(self.image)Woodlands Cres, Browns Bay
        visres, comp, residual = spectral_line_imaging(self.vis, model, continuum_model=model,
                                                       deconvolve_spectral=False)
        export_image_to_fits(comp, "%s/test_pipelines-spectral-no-deconvolution-imaging-comp.fits" % (self.dir))

    def test_spectral_line_imaging_with_deconvolution(self):
        model = create_empty_image_like(self.image)
        visres, comp, residual = spectral_line_imaging(self.vis, model, continuum_model=self.image, algorithm='hogbom',
                                                       deconvolve_spectral=True)
        export_image_to_fits(comp, "%s/test_pipelines-spectral-with-deconvolution-imaging-comp.fits" % (self.dir))

    def test_fast_imaging(self):
        fipipe = fast_imaging(vis=self.vis, components=self.comp, Gsolinit=300.0)
    
    def test_EOR(self):
        eorpipe = eor(vis=self.vis, components=self.comp, Gsolinit=300.0)
    """

if __name__ == '__main__':
    unittest.main()
