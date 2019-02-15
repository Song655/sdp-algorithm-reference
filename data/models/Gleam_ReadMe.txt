VIII/100   GaLactic and Extragalactic All-sky MWA survey  (Hurley-Walker+, 2016)
================================================================================
GaLactic and Extragalactic All-sky Murchison Wide Field Array (GLEAM) survey.
I: A low-frequency extragalactic catalogue.
    Hurley-Walker N., Callingham J.R., Hancock P.J., Franzen T.M.O.,
    Hindson L., Kapinska A.D., Morgan J., Offringa A.R., Wayth R.B., Wu C.,
    Zheng Q., Murphy T., Bell M.E., Dwarakanath K.S., For B., Gaensler B.M.,
    Johnston-Hollitt M., Lenc E., Procopio P., Staveley-Smith L., Ekers R.,
    Bowman J.D., Briggs F., Cappallo R.J., Deshpande A.A., Greenhill L.,
    Hazelton B.J., Kaplan D.L., Lonsdale C.J., McWhirter S.R., Mitchell D.A.,
    Morales M.F., Morgan E., Oberoi D., Ord S.M., Prabu T., Udaya Shankar N.,
    Srivani K.S., Subrahmanyan R., Tingay S.J., Webster R.L., Williams A.,
    Williams C.L.
   <Mon. Not. R. Astron. Soc., 464, 1146-1167 (2017)>
   =2017MNRAS.464.1146H
   =2016yCat.8100....0H
================================================================================
ADC_Keywords: Surveys ; Radio sources ; Galaxy catalogs ; Morphology
Keywords: techniques: interferometric - galaxies: general -
          radio continuum: surveys

Abstract:
    Using the Murchison Widefield Array (MWA), the low-frequency Square
    Kilometre Array (SKA1 LOW) precursor located in Western Australia,
    we have completed the GaLactic and Extragalactic All-sky MWA (GLEAM)
    survey, and present the resulting extragalactic catalogue, utilising
    the first year of observations. The catalogue covers 24,402 square
    degrees, over declinations south of +30{deg} and Galactic latitudes
    outside 10{deg} of the Galactic plane, excluding some areas such as
    the Magellanic Clouds. It contains 307,456 radio sources with 20
    separate flux density measurements across 72-231MHz, selected from a
    time- and frequency- integrated image centred at 200MHz, with a
    resolution of ~=2'. Over the catalogued region, we estimate that the
    catalogue is 90% complete at 170mJy, and 50% complete at 55mJy, and
    large areas are complete at even lower flux density levels. Its
    reliability is 99.97% above the detection threshold of 5{sigma}, which
    itself is typically 50mJy. These observations constitute the widest
    fractional bandwidth and largest sky area survey at radio frequencies
    to date, and calibrate the low frequency flux density scale of the
    southern sky to better than 10%. This paper presents details of the
    flagging, imaging, mosaicking, and source extraction/characterisation,
    as well as estimates of the completeness and reliability. All source
    measurements and images are available online. This is the first in a
    series of publications describing the GLEAM survey results.

Description:
    This paper concerns only data collected in the first year, i.e. four
    weeks between June 2013 and July 2014. We also do not image every
    observation, since the survey is redundant across approximately 50% of
    the observed RA ranges, and some parts are adversely acted by the
    Galactic plane and Centaurus A. Table 1 lists the observations which
    have been used to create this first GLEAM catalogue.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat        39       28   GLEAM first year observing parameters
gleamegc.dat    3152   307455   GLEAM EGC catalog
GLEAM_EGC.fits  2880   136178   FITS version of the catalog
GLEAM_EGC.vot    120  96234376  VOtable version of the catalog
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 10  A10   "date"  Obs.Date  Observation date
  12- 15  F4.1  h       RA1       RA range
      16  A1    ---     ---       [-]
  17- 20  F4.1  h       RA2       RA range
  22- 24  I3    deg     DE        Declination
  26- 27  I2    ---     Nflag     Number of flagged tiles out of the 128
                                   available
  29- 39  A11   ---     Cal       Calibrator (1)
--------------------------------------------------------------------------------
Note (1): The calibrator is used to find initial bandpass and phase corrections
  as described in Section 2.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: gleamegc.dat
--------------------------------------------------------------------------------
     Bytes Format Units     Label      Explanations
--------------------------------------------------------------------------------
    1-   5  A5    ---       ---        [GLEAM]
    7-  20  A14   ---       GLEAM      GLEAM name (JHHMMSS+DDMMSS) (Name)
   22-  30  F9.6  Jy/beam   bckwide    ? Background level in wide (170-231MHz)
                                        image (background_wide)
   32-  39  F8.6  Jy/beam   lrmswide   ? Local noise leve in wide (170-231MHz)
                                        image (local_rms_wide)
   41-  42  I2    h         RAh        Right Ascension J2000 (ra_str)
   44-  45  I2    min       RAm        Right Ascension J2000 (ra_str)
   47-  51  F5.2  s         RAs        [0/60] Right Ascension J2000 (ra_str)
        53  A1    ---       DE-        Declination sign J2000 (dec_str)
   54-  55  I2    deg       DEd        Declination J2000 (dec_str)
   57-  58  I2    arcmin    DEm        Declination J2000 (dec_str)
   60-  64  F5.2  arcsec    DEs        [0/60] Declination J2000 (dec_str)
   66-  75  F10.6 deg       RAdeg      Right Ascension J2000 (RAJ2000)
   79-  86  F8.6  deg     e_RAdeg      ?=- rms uncertainty on RAdeg
                                        (err_RAJ2000) (1)
   88-  97  F10.6 deg       DEdeg      Declination J2000 (DEJ2000)
  101- 108  F8.6  deg     e_DEdeg      ?=- rms uncertainty on DEdeg
                                       (err_DEJ2000) (1)
  110- 119  F10.6 Jy/beam   Fpwide     Peak flux in wide (170-231MHz) image
                                        (peak_flux_wide)
  121- 128  F8.6  Jy/beam e_Fpwide     rms uncertainty in fit for peak flux in
                                        wide image (err_peak_flux_wide)
  130- 139  F10.6 Jy        Fintwide   Integrated flux in wide (170-231MHz)
                                        image (int_flux_wide)
  141- 152  E12.6 Jy      e_Fintwide   rms uncertainty in fit for integrated
                                        flux in wide image (e_Fintwide)
  154- 165  E12.6 arcsec    awide      Fitted semi-major axis in wide
                                        (170-231MHz) image (a_wide)
  167- 178  E12.6 arcsec  e_awide      ?=-1 rms uncertainty in fitted semi-major
                                        axis in wide image (err_a_wide)
  180- 187  F8.4  arcsec    bwide      Fitted semi-minor axis in wide
                                        (170-231MHz) image (b_wide)
  189- 199  F11.6 arcsec  e_bwide      ?=-1 rms uncertainty in fitted semi-minor
                                        axis in wide image (err_b_wide)
  201- 210  F10.6 deg       pawide     Fitted position angle in wide
                                        (170-231MHz) image (pa_wide)
  212- 222  F11.6 deg     e_pawide     ?=-1 rms uncertainty in fitted position
                                        angle in wide image (err_pa_wide)
  224- 232  F9.6  Jy/beam   resmwide   Mean value of data-model in wide
                                        (170-231MHz) image (residual_mean_wide)
  234- 241  F8.6  Jy/beam   resstdwide Standard deviation of data-model in wide
                                        (170-231MHz) image (residual_std_wide)
  243- 244  I2    %         eabsFpct   Percent error in absolute flux scale -
                                        all frequencies (err_abs_flux_pct)
       246  I1    %         efitFpct   Percent error on internal flux scale -
                                        all frequencies (err_fit_flux_pct)
  248- 254  F7.3  arcsec    psfawide   Semi-major axis of the point spread
                                        function in wide (170-231MHz) image
                                        (psf_a_wide)
  256- 262  F7.3  arcsec    psfbwide   Semi-minor axis of the point spread
                                        function in wide (170-231MHz) image
                                        (psf_b_wide)
  264- 273  F10.6 deg       psfPAwide  Position angle of the point spread
                                        function in wide (170-231MHz) image
                                        (psf_pa_wide)
  275- 283  F9.6  Jy/beam   bck076     ? Background level in 072-080MHz image
                                        (background_076)
  285- 292  F8.6  Jy/beam   lrms076    ? Local noise level in 072-080MHz image
                                        (local_rms_076)
  294- 304  F11.6 Jy/beam   Fp076      ? Peak flux in 072-080MHz image
                                        (peak_flux_076)
  306- 313  F8.6  Jy/beam e_Fp076      ? rms uncertainty in fit for peak flux in
                                        072-080MHz image (err_peak_flux_076)
  315- 325  F11.6 Jy        Fint076    ? Integrated flux in 072-080MHz image
                                        (int_flux_076)
  327- 334  F8.6  Jy      e_Fint076    ? rms uncertainty in fit for integrated
                                        flux in 072-080MHz image
                                        (err_int_flux_076)
  336- 347  E12.6 arcsec    a076       ? Fitted semi-major axis in 072-080MHz
                                        image (a_076)
  349- 355  F7.3  arcsec    b076       ? Fitted semi-minor axis in 072-080MHz
                                        image (b_076)
  357- 366  F10.6 deg       pa076      ? Fitted position angle in 072-080MHz
                                        image (pa_076)
  368- 376  F9.6  Jy/beam   resm076    ? Mean value of data-model in 072-080MHz
                                        image (residual_mean_076)
  378- 386  F9.6  Jy/beam   resstd076  ? Standard deviation of data-model in
                                        072-080MHz image (residual_std_076)
  388- 394  F7.3  arcsec    psfa076    ? Semi-major axis of the point spread
                                        function in 072-080MHz image (psf_a_076)
  396- 402  F7.3  arcsec    psfb076    ? Semi-minor axis of the point spread
                                        function in 072-080MHz image (psf_b_076)
  404- 413  F10.6 deg       psfPA076   ? Position angle of the point spread
                                        function in 072-080MHz image
                                        (psf_pa_076)
  415- 423  F9.6  Jy/beam   bck084     ? Background level in 080-088MHz image
                                        (background_084)
  425- 432  F8.6  Jy/beam   lrms084    ? Local noise level in 080-088MHz image
                                        (local_rms_084)
  434- 444  F11.6 Jy/beam   Fp084      ? Peak flux in 080-088MHz image
                                        (peak_flux_084)
  446- 454  F9.6  Jy/beam e_Fp084      ? rms uncertainty in fit for peak flux in
                                        080-088MHz image (err_peak_flux_084)
  456- 466  F11.6 Jy        Fint084    ? Integrated flux in 080-088MHz image
                                        (int_flux_084)
  468- 476  F9.6  Jy      e_Fint084    ? rms uncertainty in fit for integrated
                                        flux in 080-088MHz image
                                        (err_int_flux_084)
  478- 489  E12.6 arcsec    a084       ? Fitted semi-major axis in 080-088MHz
                                        image (a_084)
  491- 497  F7.3  arcsec    b084       ? Fitted semi-minor axis in 080-088MHz
                                        image (b_084)
  499- 508  F10.6 deg       pa084      ? Fitted position angle in 080-088MHz
                                        image (pa_084)
  510- 518  F9.6  Jy/beam   resm084    ? Mean value of data-model in 080-088MHz
                                        image (residual_mean_084)
  520- 527  F8.6  Jy/beam   resstd084  ? Standard deviation of data-model in
                                        080-088MHz image (residual_std_084)
  529- 535  F7.3  arcsec    psfa084    ? Semi-major axis of the point spread
                                        function in 080-088MHz image (psf_a_084)
  537- 543  F7.3  arcsec    psfb084    ? Semi-minor axis of the point spread
                                        function in 080-088MHz image (psf_b_084)
  545- 554  F10.6 deg       psfPA084   ? Position angle of the point spread
                                       function in 080-088MHz image (psf_pa_084)
  556- 564  F9.6  Jy/beam   bck092     ? Background level in 088-095MHz image
                                        (background_092)
  566- 573  F8.6  Jy/beam   lrms092    ? Local noise level in 088-095MHz image
                                        (local_rms_092)
  575- 585  F11.6 Jy/beam   Fp092      ? Peak flux in 088-095MHz image
                                        (peak_flux_092)
  587- 595  F9.6  Jy/beam e_Fp092      ? rms uncertainty in fit for peak flux in
                                        088-095MHz image (err_peak_flux_092)
  597- 607  F11.6 Jy        Fint092    ? Integrated flux in 088-095MHz image
                                        (int_flux_092)
  609- 617  F9.6  Jy      e_Fint092    ? rms uncertainty in fit for integrated
                                        flux in 088-095MHz image
                                        (err_int_flux_092)
  619- 630  E12.6 arcsec    a092       ? Fitted semi-major axis in 088-095MHz
                                        image (a_092)
  632- 638  F7.3  arcsec    b092       ? Fitted semi-minor axis in 088-095MHz
                                        image (b_092)
  640- 649  F10.6 deg       pa092      ? Fitted position angle in 088-095MHz
                                        image (pa_092)
  651- 659  F9.6  Jy/beam   resm092    ? Mean value of data-model in 088-095MHz
                                        image (residual_mean_092)
  661- 668  F8.6  Jy/beam   resstd092  ? Standard deviation of data-model in
                                        088-095MHz image (residual_std_092)
  670- 676  F7.3  arcsec    psfa092    ? Semi-major axis of the point spread
                                        function in 088-095MHz image (psf_a_092)
  678- 684  F7.3  arcsec    psfb092    ? Semi-minor axis of the point spread
                                        function in 088-095MHz image (psf_b_092)
  686- 695  F10.6 deg       psfPA092   ? Position angle of the point spread
                                       function in 088-095MHz image (psf_pa_092)
  697- 705  F9.6  Jy/beam   bck099     ? Background level in 095-103MHz image
                                        (background_099)
  707- 714  F8.6  Jy/beam   lrms099    ? Local noise level in 095-103MHz image
                                        (local_rms_099)
  716- 726  F11.6 Jy/beam   Fp099      ? Peak flux in 095-103MHz image
                                        (peak_flux_099)
  728- 736  F9.6  Jy/beam e_Fp099      ? rms uncertainty in fit for peak flux in
                                        095-103MHz image (err_peak_flux_099)
  738- 748  F11.6 Jy        Fint099    ? Integrated flux in 095-103MHz image
                                        (int_flux_099)
  750- 758  F9.6  Jy      e_Fint099    ? rms uncertainty in fit for integrated
                                        flux in 095-103MHz image
                                        (err_int_flux_099)
  760- 771  E12.6 arcsec    a099       ? Fitted semi-major axis in 095-103MHz
                                        image (a_099)
  773- 779  F7.3  arcsec    b099       ? Fitted semi-minor axis in 095-103MHz
                                        image (b_099)
  781- 790  F10.6 deg       pa099      ? Fitted position angle in 095-103MHz
                                        image (pa_099)
  792- 800  F9.6  Jy/beam   resm099    ? Mean value of data-model in 095-103MHz
                                        image (residual_mean_099)
  802- 810  F9.6  Jy/beam   resstd099  ? Standard deviation of data-model in
                                        095-103MHz image (residual_std_099)
  812- 818  F7.3  arcsec    psfa099    ? Semi-major axis of the point spread
                                        function in 095-103MHz image (psf_a_099)
  820- 826  F7.3  arcsec    psfb099    ? Semi-minor axis of the point spread
                                        function in 095-103MHz image (psf_b_099)
  828- 837  F10.6 deg       psfPA099   ? Position angle of the point spread
                                       function in 095-103MHz image (psf_pa_099)
  839- 847  F9.6  Jy/beam   bck107     ? Background level in 103-111MHz image
                                        (background_107)
  849- 856  F8.6  Jy/beam   lrms107    ? Local noise level in 103-111MHz image
                                        (local_rms_107)
  858- 868  F11.6 Jy/beam   Fp107      ? Peak flux in 103-111MHz image
                                        (peak_flux_107)
  870- 878  F9.6  Jy/beam e_Fp107      ? rms uncertainty in fit for peak flux in
                                        103-111MHz image (err_peak_flux_107)
  880- 890  F11.6 Jy        Fint107    ? Integrated flux in 103-111MHz image
                                        (int_flux_107)
  892- 900  F9.6  Jy      e_Fint107    ? rms uncertainty in fit for integrated
                                        flux in 103-111MHz image
                                        (err_int_flux_107)
  902- 913  E12.6 arcsec    a107       ? Fitted semi-major axis in 103-111MHz
                                        image (a_107)
  915- 921  F7.3  arcsec    b107       ? Fitted semi-minor axis in 103-111MHz
                                        image (b_107)
  923- 932  F10.6 deg       pa107      ? Fitted position angle in 103-111MHz
                                        image (pa_107)
  934- 942  F9.6  Jy/beam   resm107    ? Mean value of data-model in 103-111MHz
                                        image (residual_mean_107)
  944- 951  F8.6  Jy/beam   resstd107  ? Standard deviation of data-model in
                                        103-111MHz image (residual_std_107)
  953- 959  F7.3  arcsec    psfa107    ? Semi-major axis of the point spread
                                        function in 103-111MHz image (psf_a_107)
  961- 967  F7.3  arcsec    psfb107    ? Semi-minor axis of the point spread
                                        function in 103-111MHz image (psf_b_107)
  969- 978  F10.6 deg       psfPA107   ? Position angle of the point spread
                                       function in 103-111MHz image (psf_pa_107)
  980- 988  F9.6  Jy/beam   bck115     ? Background level in 111-118MHz image
                                        (background_115)
  990- 997  F8.6  Jy/beam   lrms115    ? Local noise level in 111-118MHz image
                                        (local_rms_115)
  999-1009  F11.6 Jy/beam   Fp115      ? Peak flux in 111-118MHz image
                                        (peak_flux_115)
 1011-1019  F9.6  Jy/beam e_Fp115      ? rms uncertainty in fit for peak flux in
                                        111-118MHz image (err_peak_flux_115)
 1021-1031  F11.6 Jy        Fint115    ? Integrated flux in 111-118MHz image
                                        (int_flux_115)
 1033-1041  F9.6  Jy      e_Fint115    ? rms uncertainty in fit for integrated
                                        flux in 111-118MHz image
                                        (err_int_flux_115)
 1043-1054  E12.6 arcsec    a115       ? Fitted semi-major axis in 111-118MHz
                                        image (a_115)
 1056-1062  F7.3  arcsec    b115       ? Fitted semi-minor axis in 111-118MHz
                                        image (b_115)
 1064-1073  F10.6 deg       pa115      ? Fitted position angle in 111-118MHz
                                        image (pa_115)
 1075-1083  F9.6  Jy/beam   resm115    ? Mean value of data-model in 111-118MHz
                                        image (residual_mean_115)
 1085-1092  F8.6  Jy/beam   resstd115  ? Standard deviation of data-model in
                                        111-118MHz image (residual_std_115)
 1094-1100  F7.3  arcsec    psfa115    ? Semi-major axis of the point spread
                                        function in 111-118MHz image (psf_a_115)
 1102-1108  F7.3  arcsec    psfb115    ? Semi-minor axis of the point spread
                                        function in 111-118MHz image (psf_b_115)
 1110-1119  F10.6 deg       psfPA115   ? Position angle of the point spread
                                       function in 111-118MHz image (psf_pa_115)
 1121-1129  F9.6  Jy/beam   bck122     ? Background level in 118-126MHz image
                                        (background_122)
 1131-1138  F8.6  Jy/beam   lrms122    ? Local noise level in 118-126MHz image
                                        (local_rms_122)
 1140-1150  F11.6 Jy/beam   Fp122      ? Peak flux in 118-126MHz image
                                        (peak_flux_122)
 1152-1160  F9.6  Jy/beam e_Fp122      ? rms uncertainty in fit for peak flux in
                                        118-126MHz image (err_peak_flux_122)
 1162-1172  F11.6 Jy        Fint122    ? Integrated flux in 118-126MHz image
                                        (int_flux_122)
 1174-1182  F9.6  Jy      e_Fint122    ? rms uncertainty in fit for integrated
                                        flux in 118-126MHz image
                                        (err_int_flux_122)
 1184-1195  E12.6 arcsec    a122       ? Fitted semi-major axis in 118-126MHz
                                        image (a_122)
 1197-1203  F7.3  arcsec    b122       ? Fitted semi-minor axis in 118-126MHz
                                        image (b_122)
 1205-1214  F10.6 deg       pa122      ? Fitted semi-minor axis in 118-126MHz
                                        image (pa_122)
 1216-1224  F9.6  Jy/beam   resm122    ? Mean value of data-model in 118-126MHz
                                        image (residual_mean_122)
 1226-1233  F8.6  Jy/beam   resstd122  ? Standard deviation of data-model in
                                        118-126MHz image (residual_std_122)
 1235-1241  F7.3  arcsec    psfa122    ? Semi-major axis of the point spread
                                        function in 118-126MHz image (psf_a_122)
 1243-1249  F7.3  arcsec    psfb122    ? Semi-minor axis of the point spread
                                        function in 118-126MHz image (psf_b_122)
 1251-1260  F10.6 deg       psfPA122   ? Position angle of the point spread
                                       function in 118-126MHz image (psf_pa_122)
 1262-1270  F9.6  Jy/beam   bck130     ? Background level in 126-134MHz image
                                        (background_130)
 1272-1279  F8.6  Jy/beam   lrms130    ? Local noise level in 126-134MHz image
                                        (local_rms_130)
 1281-1291  F11.6 Jy/beam   Fp130      ? Peak flux in 126-134MHz image
                                        (peak_flux_130)
 1293-1301  F9.6  Jy/beam e_Fp130      ? rms uncertainty in fit for peak flux in
                                        126-134MHz image (err_peak_flux_130)
 1303-1313  F11.6 Jy        Fint130    ? Integrated flux in 126-134MHz image
                                        (int_flux_130)
 1315-1323  F9.6  Jy      e_Fint130    ? rms uncertainty in fit for integrated
                                        flux in 126-134MHz image
                                        (err_int_flux_130)
 1325-1336  E12.6 arcsec    a130       ? Fitted semi-major axis in 126-134MHz
                                        image (a_130)
 1338-1344  F7.3  arcsec    b130       ? Fitted semi-minor axis in 126-134MHz
                                        image (b_130)
 1346-1355  F10.6 deg       pa130      ? Fitted position angle in 126-134MHz
                                        image (pa_130)
 1357-1365  F9.6  Jy/beam   resm130    ? Mean value of data-model in 126-134MHz
                                        image (residual_mean_130)
 1367-1374  F8.6  Jy/beam   resstd130  ? Standard deviation of data-model in
                                        126-134MHz image (residual_std_130)
 1376-1382  F7.3  arcsec    psfa130    ? Semi-major axis of the point spread
                                        function in 126-134MHz image (psf_a_130)
 1384-1390  F7.3  arcsec    psfb130    ? Semi-minor axis of the point spread
                                        function in 126-134MHz image (psf_b_130)
 1392-1401  F10.6 deg       psfPA130   ? Position angle of the point spread
                                       function in 126-134MHz image (psf_pa_130)
 1403-1411  F9.6  Jy/beam   bck143     ? Background level in 139-147MHz image
                                        (background_143)
 1413-1420  F8.6  Jy/beam   lrms143    ? Local noise level in 139-147MHz image
                                        (local_rms_143)
 1422-1432  F11.6 Jy/beam   Fp143      ? Peak flux in 139-147MHz image
                                        (peak_flux_143)
 1434-1442  F9.6  Jy/beam e_Fp143      ? rms uncertainty in fit for peak flux
                                        in 139-147MHz image (err_peak_flux_143)
 1444-1454  F11.6 Jy        Fint143    ? Integrated flux in 139-147MHz image
                                        (int_flux_143)
 1456-1464  F9.6  Jy      e_Fint143    ? rms uncertainty in fit for integrated
                                        flux in 139-147MHz image
                                        (err_int_flux_143)
 1466-1477  E12.6 arcsec    a143       ? Fitted semi-major axis in 139-147MHz
                                        image (a_143)
 1479-1485  F7.3  arcsec    b143       ? Fitted semi-minor axis in 139-147MHz
                                        image (b_143)
 1487-1496  F10.6 deg       pa143      ? Fitted position angle in 139-147MHz
                                        image (pa_143)
 1498-1506  F9.6  Jy/beam   resm143    ? Mean value of data-model in 139-147MHz
                                        image (residual_mean_143)
 1508-1515  F8.6  Jy/beam   resstd143  ? Standard deviation of data-model in
                                        139-147MHz image (residual_std_143)
 1517-1523  F7.3  arcsec    psfa143    ? Semi-major axis of the point spread
                                        function in 139-147MHz image (psf_a_143)
 1525-1531  F7.3  arcsec    psfb143    ? Semi-minor axis of the point spread
                                        function in 139-147MHz image (psf_b_143)
 1533-1542  F10.6 deg       psfPA143   ? Position angle of the point spread
                                       function in 139-147MHz image (psf_pa_143)
 1544-1552  F9.6  Jy/beam   bck151     ? Background level in 147-154MHz image
                                        (background_151)
 1554-1561  F8.6  Jy/beam   lrms151    ? Local noise level in 147-154MHz image
                                        (local_rms_151)
 1563-1573  F11.6 Jy/beam   Fp151      ? Peak flux in 147-154MHz image
                                        (peak_flux_151)
 1575-1583  F9.6  Jy/beam e_Fp151      ? rms uncertainty in fit for peak flux
                                        in 147-154MHz image (err_peak_flux_151)
 1585-1595  F11.6 Jy        Fint151    ? Integrated flux in 147-154MHz image
                                        (int_flux_151)
 1597-1605  F9.6  Jy      e_Fint151    ? rms uncertainty in fit for integrated
                                        flux in 147-154MHz image
                                        (err_int_flux_151)
 1607-1618  E12.6 arcsec    a151       ? Fitted semi-major axis in 147-154MHz
                                        image (a_151)
 1620-1626  F7.3  arcsec    b151       ? Fitted semi-minor axis in 147-154MHz
                                        image (b_151)
 1628-1637  F10.6 deg       pa151      ? Fitted position angle in 147-154MHz
                                        image (pa_151)
 1639-1647  F9.6  Jy/beam   resm151    ? Mean value of data-model in 147-154MHz
                                        image (residual_mean_151)
 1649-1656  F8.6  Jy/beam   resstd151  ? Standard deviation of data-model in
                                        147-154MHz image (residual_std_151)
 1658-1664  F7.3  arcsec    psfa151    ? Semi-major axis of the point spread
                                        function in 147-154MHz image (psf_a_151)
 1666-1672  F7.3  arcsec    psfb151    ? Semi-minor axis of the point spread
                                        function in 147-154MHz image (psf_b_151)
 1674-1683  F10.6 deg       psfPA151   ? Position angle of the point spread
                                       function in 147-154MHz image (psf_pa_151)
 1685-1693  F9.6  Jy/beam   bck158     ? Background level in 154-162MHz image
                                        (background_158)
 1695-1702  F8.6  Jy/beam   lrms158    ? Local noise level in 154-162MHz image
                                        (local_rms_158)
 1704-1714  F11.6 Jy/beam   Fp158      ? Peak flux in 154-162MHz image
                                        (peak_flux_158)
 1716-1724  F9.6  Jy/beam e_Fp158      ? rms uncertainty in fit for peak flux in
                                        154-162MHz image (err_peak_flux_158)
 1726-1736  F11.6 Jy        Fint158    ? Integrated flux in 154-162MHz image
                                        (int_flux_158)
 1738-1746  F9.6  Jy      e_Fint158    ? rms uncertainty in fit for integrated
                                        flux in 154-162MHz image
                                        (err_int_flux_158)
 1748-1759  E12.6 arcsec    a158       ? Fitted semi-major axis in 154-162MHz
                                        image (a_158)
 1761-1767  F7.3  arcsec    b158       ? Fitted semi-minor axis in 154-162MHz
                                        image (b_158)
 1769-1778  F10.6 deg       pa158      ? Fitted position angle in 154-162MHz
                                        image (pa_158)
 1780-1788  F9.6  Jy/beam   resm158    ? Mean value of data-model in 154-162MHz
                                        image (residual_mean_158)
 1790-1797  F8.6  Jy/beam   resstd158  ? Standard deviation of data-model in
                                        154-162MHz image (residual_std_158)
 1799-1805  F7.3  arcsec    psfa158    ? Semi-major axis of the point spread
                                        function in 154-162MHz image (psf_a_158)
 1807-1813  F7.3  arcsec    psfb158    ? Semi-minor axis of the point spread
                                        function in 154-162MHz image (psf_b_158)
 1815-1824  F10.6 deg       psfPA158   ? Position angle of the point spread
                                       function in 154-162MHz image (psf_pa_158)
 1826-1834  F9.6  Jy/beam   bck166     ? Background level in 162-170MHz image
                                        (background_166)
 1836-1843  F8.6  Jy/beam   lrms166    ?  Local noise level in 162-170MHz image
                                        (local_rms_166)
 1845-1855  F11.6 Jy/beam   Fp166      ? Peak flux in 162-170MHz image
                                        (peak_flux_166)
 1857-1865  F9.6  Jy/beam e_Fp166      ? rms uncertainty in fit for peak flux in
                                        162-170MHz image (err_peak_flux_166)
 1867-1877  F11.6 Jy        Fint166    ? Integrated flux in 162-170MHz image
                                        (int_flux_166)
 1879-1887  F9.6  Jy      e_Fint166    ? rms uncertainty in fit for integrated
                                        flux in 162-170MHz image
                                        (err_int_flux_166)
 1889-1900  E12.6 arcsec    a166       ? Fitted semi-major axis in 162-170MHz
                                        image (a_166)
 1902-1908  F7.3  arcsec    b166       ? Fitted semi-minor axis in 162-170MHz
                                        image (b_166)
 1910-1919  F10.6 deg       pa166      ? Fitted position angle in 162-170MHz
                                        image (pa_166)
 1921-1929  F9.6  Jy/beam   resm166    ? Mean value of data-model in 162-170MHz
                                        image (residual_mean_166)
 1931-1938  F8.6  Jy/beam   resstd166  ? Standard deviation of data-model in
                                        162-170MHz image (residual_std_166)
 1940-1946  F7.3  arcsec    psfa166    ? Semi-major axis of the point spread
                                        function in 162-170MHz image (psf_a_166)
 1948-1954  F7.3  arcsec    psfb166     ? Semi-minor axis of the point spread
                                       function in 162-170MHz image (psf_b_166)
 1956-1965  F10.6 deg       psfPA166   ? Position angle of the point spread
                                       function in 162-170MHz image (psf_pa_166)
 1967-1975  F9.6  Jy/beam   bck174     ? Background level in 170-177MHz image
                                        (background_174)
 1977-1984  F8.6  Jy/beam   lrms174    ? Local noise level in 170-177MHz image
                                        (local_rms_174)
 1986-1996  F11.6 Jy/beam   Fp174      ? Peak flux in 170-177MHz image
                                        (peak_flux_174)
 1998-2006  F9.6  Jy/beam e_Fp174      ? rms uncertainty in fit for peak flux in
                                        170-177MHz image (err_peak_flux_174)
 2008-2018  F11.6 Jy        Fint174    ? Integrated flux in 170-177MHz image
                                        (int_flux_174)
 2020-2028  F9.6  Jy      e_Fint174    ? rms uncertainty in fit for integrated
                                        flux in 170-177MHz image
                                        (err_int_flux_174)
 2030-2041  E12.6 arcsec    a174       ? Fitted semi-major axis in 170-177MHz
                                        image (a_174)
 2043-2049  F7.3  arcsec    b174       ? Fitted semi-minor axis in 170-177MHz
                                        image (b_174)
 2051-2060  F10.6 deg       pa174      ? Fitted position angle in 170-177MHz
                                        image (pa_174)
 2062-2070  F9.6  Jy/beam   resm174    ? Mean value of data-model in 170-177MHz
                                        image (residual_mean_174)
 2072-2079  F8.6  Jy/beam   resstd174  ? Standard deviation of data-model in
                                        170-177MHz image (residual_std_174)
 2081-2087  F7.3  arcsec    psfa174    ? Semi-major axis of the point spread
                                        function in 170-177MHz image (psf_a_174)
 2089-2095  F7.3  arcsec    psfb174    ? Semi-minor axis of the point spread
                                        function in 170-177MHz image (psf_b_174)
 2097-2106  F10.6 deg       psfPA174   ? Position angle of the point spread
                                       function in 170-177MHz image (psf_pa_174)
 2108-2116  F9.6  Jy/beam   bck181     ? Background level in 177-185MHz image
                                        (background_181)
 2118-2125  F8.6  Jy/beam   lrms181    ? Local noise level in 177-185MHz image
                                        (local_rms_181)
 2127-2137  F11.6 Jy/beam   Fp181      ? Peak flux in 177-185MHz image
                                        (peak_flux_181)
 2139-2147  F9.6  Jy/beam e_Fp181      ? rms uncertainty in fit for peak flux in
                                        177-185MHz image (err_peak_flux_181)
 2149-2159  F11.6 Jy        Fint181    ? Integrated flux in 177-185MHz image
                                        (int_flux_181)
 2161-2169  F9.6  Jy      e_Fint181    ? rms uncertainty in fit for integrated
                                        flux in 177-185MHz image
                                        (err_int_flux_181)
 2171-2182  E12.6 arcsec    a181       ? Fitted semi-major axis in 177-185MHz
                                        image (a_181)
 2184-2191  F8.4  arcsec    b181       ? Fitted semi-minor axis in 177-185MHz
                                        image (b_181)
 2193-2202  F10.6 deg       pa181      ? Fitted position angle in 177-185MHz
                                        image (pa_181)
 2204-2212  F9.6  Jy/beam   resm181    ? Mean value of data-model in 177-185MHz
                                        image (residual_mean_181)
 2214-2221  F8.6  Jy/beam   resstd181  ? Standard deviation of data-model in
                                        177-185MHz image (residual_std_181)
 2223-2229  F7.3  arcsec    psfa181    ? Semi-major axis of the point spread
                                        function in 177-185MHz image (psf_a_181)
 2231-2237  F7.3  arcsec    psfb181    ? Semi-minor axis of the point spread
                                        function in 177-185MHz image (psf_b_181)
 2239-2248  F10.6 deg       psfPA181   ? Position angle of the point spread
                                       function in 177-185MHz image (psf_pa_181)
 2250-2258  F9.6  Jy/beam   bck189     ? Background level in 185-193MHz image
                                        (background_189)
 2260-2267  F8.6  Jy/beam   lrms189    ? Local noise level in 185-193MHz image
                                        (local_rms_189)
 2269-2279  F11.6 Jy/beam   Fp189      ? Peak flux in 185-193MHz image
                                        (peak_flux_189)
 2281-2289  F9.6  Jy/beam e_Fp189      ? rms uncertainty in fit for peak flux in
                                        185-193MHz image (err_peak_flux_189)
 2291-2301  F11.6 Jy        Fint189    ? Integrated flux in 185-193MHz image
                                        (int_flux_189)
 2303-2311  F9.6  Jy      e_Fint189    ? rms uncertainty in fit for integrated
                                        flux in 185-193MHz image
                                        (err_int_flux_189)
 2313-2324  E12.6 arcsec    a189       ? Fitted semi-major axis in 185-193MHz
                                        image (a_189)
 2326-2333  F8.4  arcsec    b189       ? Fitted semi-minor axis in 185-193MHz
                                        image (b_189)
 2335-2344  F10.6 deg       pa189      ? Fitted position angle in 185-193MHz
                                        image (pa_189)
 2346-2354  F9.6  Jy/beam   resm189    ? Mean value of data-model in 185-193MHz
                                        image (residual_mean_189)
 2356-2363  F8.6  Jy/beam   resstd189  ? Standard deviation of data-model in
                                        185-193MHz image (residual_std_189)
 2365-2371  F7.3  arcsec    psfa189    ? Semi-major axis of the point spread
                                        function in 185-193MHz image (psf_a_189)
 2373-2379  F7.3  arcsec    psfb189    ? Semi-minor axis of the point spread
                                        function in 185-193MHz image (psf_b_189)
 2381-2390  F10.6 deg       psfPA189   ? Position angle of the point spread
                                       function in 185-193MHz image (psf_pa_189)
 2392-2400  F9.6  Jy/beam   bck197     ? Background level in 193-200MHz image
                                        (background_197)
 2402-2409  F8.6  Jy/beam   lrms197    ? Local noise level in 193-200MHz image
                                        (local_rms_197)
 2411-2421  F11.6 Jy/beam   Fp197      ? Peak flux in 193-200MHz image
                                        (peak_flux_197)
 2423-2431  F9.6  Jy/beam e_Fp197      ? rms uncertainty in fit for peak flux in
                                        193-200MHz image (err_peak_flux_197)
 2433-2443  F11.6 Jy        Fint197    ? Integrated flux in 193-200MHz image
                                        (int_flux_197)
 2445-2453  F9.6  Jy      e_Fint197    ? rms uncertainty in fit for integrated
                                        flux in 193-200MHz image
                                        (err_int_flux_197)
 2455-2466  E12.6 arcsec    a197       ? Fitted semi-major axis in 193-200MHz
                                        image (a_197)
 2468-2475  F8.4  arcsec    b197       ? Fitted semi-minor axis in 193-200MHz
                                        image (b_197)
 2477-2486  F10.6 deg       pa197      ? Fitted position angle in 193-200MHz
                                        image (pa_197)
 2488-2496  F9.6  Jy/beam   resm197    ? Mean value of data-model in 193-200MHz
                                        image (residual_mean_197)
 2498-2505  F8.6  Jy/beam   resstd197  ? Standard deviation of data-model in
                                        193-200MHz image (residual_std_197)
 2507-2513  F7.3  arcsec    psfa197    ? Semi-major axis of the point spread
                                        function in 193-200MHz image (psf_a_197)
 2515-2521  F7.3  arcsec    psfb197    ? Semi-minor axis of the point spread
                                        function in 193-200MHz image (psf_b_197)
 2523-2532  F10.6 deg       psfPA197   ? Position angle of the point spread
                                       function in 193-200MHz image (psf_pa_197)
 2534-2542  F9.6  Jy/beam   bck204     ? Background level in 200-208MHz image
                                        (background_204)
 2544-2551  F8.6  Jy/beam   lrms204    ? Local noise level in 200-208MHz image
                                        (local_rms_204)
 2553-2563  F11.6 Jy/beam   Fp204      ? Peak flux in 200-208MHz image
                                        (peak_flux_204)
 2565-2573  F9.6  Jy/beam e_Fp204      ? rms uncertainty in fit for peak flux in
                                        200-208MHz image (err_peak_flux_204)
 2575-2585  F11.6 Jy        Fint204    ? Integrated flux in 200-208MHz image
                                        (int_flux_204)
 2587-2595  F9.6  Jy      e_Fint204    ? rms uncertainty in fit for integrated
                                        flux in 200-208MHz image
                                        (err_int_flux_204)
 2597-2608  E12.6 arcsec    a204       ? Fitted semi-major axis in 200-208MHz
                                        image (a_204)
 2610-2617  F8.4  arcsec    b204       ? Fitted semi-minor axis in 200-208MHz
                                        image (b_204)
 2619-2628  F10.6 deg       pa204      ? Fitted position angle in 200-208MHz
                                        image (pa_204)
 2630-2638  F9.6  Jy/beam   resm204    ? Mean value of data-model in 200-208MHz
                                        image (residual_mean_204)
 2640-2647  F8.6  Jy/beam   resstd204  ? Standard deviation of data-model in
                                        200-208MHz image (residual_std_204)
 2649-2655  F7.3  arcsec    psfa204    ? Semi-major axis of the point spread
                                        function in 200-208MHz image (psf_a_204)
 2657-2663  F7.3  arcsec    psfb204    ? Semi-minor axis of the point spread
                                        function in 200-208MHz image (psf_b_204)
 2665-2674  F10.6 deg       psfPA204   ? Position angle of the point spread
                                       function in 200-208MHz image (psf_pa_204)
 2676-2684  F9.6  Jy/beam   bck212     ? Background level in 208-216MHz image
                                        (background_212)
 2686-2693  F8.6  Jy/beam   lrms212    ? Local noise level in 208-216MHz image
                                        (local_rms_212)
 2695-2705  F11.6 Jy/beam   Fp212      ? Peak flux in 208-216MHz image
                                        (peak_flux_212)
 2707-2715  F9.6  Jy/beam e_Fp212      ? rms uncertainty in fit for peak flux in
                                        208-216MHz image (err_peak_flux_212)
 2717-2727  F11.6 Jy        Fint212    ? Integrated flux in 208-216MHz image
                                        (int_flux_212)
 2729-2737  F9.6  Jy      e_Fint212    ? rms uncertainty in fit for integrated
                                        flux in 208-216MHz image
                                        (err_int_flux_212)
 2739-2750  E12.6 arcsec    a212       ? Fitted semi-major axis in 208-216MHz
                                        image (a_212)
 2752-2759  F8.4  arcsec    b212       ? Fitted semi-minor axis in 208-216MHz
                                        image (b_212)
 2761-2770  F10.6 deg       pa212      ? Fitted position angle in 208-216MHz
                                        image (pa_212)
 2772-2780  F9.6  Jy/beam   resm212    ? Mean value of data-model in 208-216MHz
                                        image (residual_mean_212)
 2782-2789  F8.6  Jy/beam   resstd212  ? Standard deviation of data-model in
                                        208-216MHz image (residual_std_212)
 2791-2797  F7.3  arcsec    psfa212    ? Semi-major axis of the point spread
                                        function in 208-216MHz image (psf_a_212)
 2799-2806  F8.4  arcsec    psfb212    ? Semi-minor axis of the point spread
                                        function in 208-216MHz image (psf_b_212)
 2808-2817  F10.6 deg       psfPA212   ? Position angle of the point spread
                                       function in 208-216MHz image (psf_pa_212)
 2819-2827  F9.6  Jy/beam   bck220     ? Background level in 216-223MHz image
                                        (background_220)
 2829-2836  F8.6  Jy/beam   lrms220    ? Local noise level in 216-223MHz image
                                        (local_rms_220)
 2838-2848  F11.6 Jy/beam   Fp220      ? Peak flux in 216-223MHz image
                                        (peak_flux_220)
 2850-2858  F9.6  Jy/beam e_Fp220      ? rms uncertainty in fit for peak flux in
                                        216-223MHz image (err_peak_flux_220)
 2860-2870  F11.6 Jy        Fint220    ? Integrated flux in 216-223MHz image
                                        (int_flux_220)
 2872-2880  F9.6  Jy      e_Fint220    ? rms uncertainty in fit for integrated
                                        flux in 216-223MHz image
                                        (err_int_flux_220)
 2882-2893  E12.6 arcsec    a220       ? Fitted semi-major axis in 216-223MHz
                                        image (a_220)
 2895-2902  F8.4  arcsec    b220       ? Fitted semi-minor axis in 216-223MHz
                                        image (b_220)
 2904-2913  F10.6 deg       pa220      ? Fitted position angle in 216-223MHz
                                        image (pa_220)
 2915-2923  F9.6  Jy/beam   resm220    ? Mean value of data-model in 216-223MHz
                                        image (residual_mean_220)
 2925-2932  F8.6  Jy/beam   resstd220  ? Standard deviation of data-model in
                                        216-223MHz image (residual_std_220)
 2934-2940  F7.3  arcsec    psfa220    ? Semi-major axis of the point spread
                                        function in 216-223MHz image (psf_a_220)
 2942-2949  F8.4  arcsec    psfb220    ? Semi-minor axis of the point spread
                                        function in 216-223MHz image (psf_b_220)
 2951-2960  F10.6 deg       psfPA220   ? Position angle of the point spread
                                       function in 216-223MHz image (psf_pa_220)
 2962-2970  F9.6  Jy/beam   bck227     ? Background level in 223-231MHz image
                                        (background_227)
 2972-2979  F8.6  Jy/beam   lrms227    ? Local noise level in 223-231MHz image
                                        (local_rms_227)
 2981-2991  F11.6 Jy/beam   Fp227      ? Peak flux in 223-231MHz image
                                        (peak_flux_227)
 2993-3001  F9.6  Jy/beam e_Fp227      ? rms uncertainty in fit for peak flux in
                                        223-231MHz image (err_peak_flux_227)
 3003-3013  F11.6 Jy        Fint227    ? Integrated flux in 223-231MHz image
                                        (int_flux_227)
 3015-3023  F9.6  Jy      e_Fint227    ? rms uncertainty in fit for integrated
                                        flux in 223-231MHz image
                                        (err_int_flux_227)
 3025-3036  E12.6 arcsec    a227       ? Fitted semi-major axis in 223-231MHz
                                        image (a_227)
 3038-3045  F8.4  arcsec    b227       ? Fitted semi-minor axis in 223-231MHz
                                        image (b_227)
 3047-3056  F10.6 deg       pa227      ? Fitted position angle in 223-231MHz
                                        image (pa_227)
 3058-3066  F9.6  Jy/beam   resm227    ? Mean value of data-model in 223-231MHz
                                        image (residual_mean_227)
 3068-3075  F8.6  Jy/beam   resstd227  ? Standard deviation of data-model in
                                        223-231MHz image (residual_std_227)
 3077-3083  F7.3  arcsec    psfa227    ? Semi-major axis of the point spread
                                        function in 223-231MHz image (psf_a_227)
 3085-3092  F8.4  arcsec    psfb227    ? Semi-minor axis of the point spread
                                        function in 223-231MHz image (psf_b_227)
 3094-3103  F10.6 deg       psfPA227   ? Position angle of the point spread
                                       function in 223-231MHz image (psf_pa_227)
 3105-3113  F9.6  ---       alpha      ? Fitted spectral index (alpha)
 3115-3123  F9.6  ---     e_alpha      ? Error on fitted spectral index
                                        (err_alpha)
 3125-3132  F8.6  ---       chi2       ? Reduced chi^2 statistic for spectral
                                        index fit (reduced_chi2)
 3134-3143  F10.6 Jy       Fintfit200  ? Fitted 200MHz integrated flux density
                                        (int_flux_fit_200)
 3145-3152  F8.6  Jy     e_Fintfit200  ? Error on fitted 200MHz integrated flux
                                        density (err_int_flux_fit_200)
--------------------------------------------------------------------------------
Note (1): No errors in positions for sources failed to fit positions correctly.
--------------------------------------------------------------------------------

Acknowledgements:
    Natasha Hurley-Walker, nhurleywalker(at)cantab.net

References:
   Wayth et al., 2016PASA...32...25W
    GLEAM: The GaLactic and Extragalactic All-sky MWA survey.

================================================================================
(End)                                        Patricia Vannier [CDS]  21-Oct-2016
