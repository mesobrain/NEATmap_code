from Environment import *
from Parameter import *

def cortex_stats(csv_path):
    
    file = open(os.path.join(csv_path, 'cell-counting.csv'))
    df = pd.read_csv(file, error_bad_lines=False)
    counts = df['count']
    ## FRP
    frp_index = np.arange(12, 22)
    frp = counts[frp_index].values
    frp_1 = (frp[0] + frp[1]) // 2
    frp_2_3 = (frp[2] + frp[3]) // 2
    frp_4 = 0
    frp_5 = (frp[4] + frp[5]) // 2
    frp_6 = int(np.mean(frp[6:]))
    frp_counts = [frp_1, frp_2_3, frp_4, frp_5, frp_6]
    ##MOp
    mop_index = np.arange(26, 36)
    mop = counts[mop_index].values
    mop_1 = (mop[0] + mop[1]) // 2
    mop_2_3 = (mop[2] + mop[3]) // 2
    mop_4 = 0
    mop_5 = (mop[4] + mop[5]) // 2
    mop_6 = int(np.mean(mop[6:]))
    mop_counts = [mop_1, mop_2_3, mop_4, mop_5, mop_6]
    ##MOs
    mos_index = np.arange(38, 48)
    mos = counts[mos_index].values
    mos_1 = (mos[0] + mos[1]) // 2
    mos_2_3 = (mos[2] + mos[3]) // 2
    mos_4 = 0
    mos_5 = (mos[4] + mos[5]) // 2
    mos_6 = int(np.mean(mos[6:]))
    mos_counts = [mos_1, mos_2_3, mos_4, mos_5, mos_6]
    ##SSp-n
    ssp_n_index = np.arange(54, 66)
    ssp_n = counts[ssp_n_index].values
    ssp_n_1 = (ssp_n[0] + ssp_n[1]) // 2
    ssp_n_2_3 = (ssp_n[2] + ssp_n[3]) // 2
    ssp_n_4 = (ssp_n[4] + ssp_n[5]) // 2
    ssp_n_5 = (ssp_n[6] + ssp_n[7]) // 2
    ssp_n_6 = int(np.mean(ssp_n[8:]))
    ssp_n_counts = [ssp_n_1, ssp_n_2_3, ssp_n_4, ssp_n_5, ssp_n_6]
    ##SSp-bfd
    ssp_bfd_index = np.arange(68, 80)
    ssp_bfd = counts[ssp_bfd_index].values
    ssp_bfd_1 = (ssp_bfd[0] + ssp_bfd[1]) // 2
    ssp_bfd_2_3 = (ssp_bfd[2] + ssp_bfd[3]) // 2
    ssp_bfd_4 = (ssp_bfd[4] + ssp_bfd[5]) // 2
    ssp_bfd_5 = (ssp_bfd[6] + ssp_bfd[7]) // 2
    ssp_bfd_6 = int(np.mean(ssp_bfd[6:]))
    ssp_bfd_counts = [ssp_bfd_1, ssp_bfd_2_3, ssp_bfd_4, ssp_bfd_5, ssp_bfd_6]
    ##SSp-ll
    ssp_ll_index = np.arange(82, 94)
    ssp_ll = counts[ssp_ll_index].values
    ssp_ll_1 = (ssp_ll[0] + ssp_ll[1]) // 2
    ssp_ll_2_3 = (ssp_ll[2] + ssp_ll[3]) // 2
    ssp_ll_4 = (ssp_ll[4] + ssp_ll[5]) // 2
    ssp_ll_5 = (ssp_ll[6] + ssp_ll[7]) // 2
    ssp_ll_6 = int(np.mean(ssp_ll[8:]))
    ssp_ll_counts = [ssp_ll_1, ssp_ll_2_3, ssp_ll_4, ssp_ll_5, ssp_ll_6]
    ##SSp-m
    ssp_m_index = np.arange(96, 108)
    ssp_m = counts[ssp_m_index].values
    ssp_m_1 = (ssp_m[0] + ssp_m[1]) // 2
    ssp_m_2_3 = (ssp_m[2] + ssp_m[3]) // 2
    ssp_m_4 = (ssp_m[4] + ssp_m[5]) // 2
    ssp_m_5 = (ssp_m[6] + ssp_m[7]) // 2
    ssp_m_6 = int(np.mean(ssp_m[8:]))
    ssp_m_counts = [ssp_m_1, ssp_m_2_3, ssp_m_4, ssp_m_5, ssp_m_6]
    ##SSp-ul
    ssp_ul_index = np.arange(110, 122)
    ssp_ul = counts[ssp_ul_index].values
    ssp_ul_1 = (ssp_ul[0] + ssp_ul[1]) // 2
    ssp_ul_2_3 = (ssp_ul[2] + ssp_ul[3]) // 2
    ssp_ul_4 = (ssp_ul[4] + ssp_ul[5]) // 2
    ssp_ul_5 = (ssp_ul[6] + ssp_ul[7]) // 2
    ssp_ul_6 = int(np.mean(ssp_ul[8:]))
    ssp_ul_counts = [ssp_ul_1, ssp_ul_2_3, ssp_ul_4, ssp_ul_5, ssp_ul_6]
    ##SSp-tr
    ssp_tr_index = np.arange(124, 136)
    ssp_tr = counts[ssp_tr_index].values
    ssp_tr_1 = (ssp_tr[0] + ssp_tr[1]) // 2
    ssp_tr_2_3 = (ssp_tr[2] + ssp_tr[3]) // 2
    ssp_tr_4 = (ssp_tr[4] + ssp_tr[5]) // 2
    ssp_tr_5 = (ssp_tr[6] + ssp_tr[7]) // 2
    ssp_tr_6 = int(np.mean(ssp_tr[8:]))
    ssp_tr_counts = [ssp_tr_1, ssp_tr_2_3, ssp_tr_4, ssp_tr_5, ssp_tr_6] 
    ##SSp-un
    ssp_un_index = np.arange(138, 150)
    ssp_un = counts[ssp_un_index].values
    ssp_un_1 = (ssp_un[0] + ssp_un[1]) // 2
    ssp_un_2_3 = (ssp_un[2] + ssp_un[3]) // 2
    ssp_un_4 = (ssp_un[4] + ssp_un[5]) // 2
    ssp_un_5 = (ssp_un[6] + ssp_un[7]) // 2
    ssp_un_6 = int(np.mean(ssp_un[8:]))
    ssp_un_counts = [ssp_un_1, ssp_un_2_3, ssp_un_4, ssp_un_5, ssp_un_6]
    ##SSs
    sss_index = np.arange(152, 164)
    sss = counts[sss_index].values
    sss_1 = (sss[0] + sss[1]) // 2
    sss_2_3 = (sss[2] + sss[3]) // 2
    sss_4 = (sss[4] + sss[5]) //2
    sss_5 = (sss[6] + sss[7]) // 2
    sss_6 = int(np.mean(sss[8:]))
    sss_count = [sss_1, sss_2_3, sss_4, sss_5, sss_6]
    ##AUDd
    audd_index = np.arange(196, 208)
    audd = counts[audd_index].values
    audd_1 = (audd[0] + audd[1]) // 2
    audd_2_3 = (audd[2] + audd[3]) // 2
    audd_4 = (audd[4] + audd[5]) // 2
    audd_5 = (audd[6] + audd[7]) // 2
    audd_6 = int(np.mean(audd[8:]))
    audd_counts = [audd_1, audd_2_3, audd_4, audd_5, audd_6]
    ##AUDp
    audp_index = np.arange(210, 222)
    audp = counts[audp_index].values
    audp_1 = (audp[0] + audp[1]) // 2
    audp_2_3 = (audp[2] + audp[3]) // 2
    audp_4 = (audp[4] + audp[5]) // 2
    audp_5 = (audp[6] + audp[7]) // 2
    audp_6 = int(np.mean(audp[8:]))
    audp_counts = [audp_1, audp_2_3, audp_4, audp_5, audp_6]
    ##AUDpo
    audpo_index = np.arange(224, 236)
    audpo = counts[audpo_index].values
    audpo_1 = (audpo[0] + audpo[1]) // 2
    audpo_2_3 = (audpo[2] + audpo[3]) // 2
    audpo_4 = (audpo[4] + audpo[5]) // 2
    audpo_5 = (audpo[6] + audpo[7]) // 2
    audpo_6 = int(np.mean(audpo[8:]))
    audpo_counts = [audpo_1, audpo_2_3, audpo_4, audpo_5, audpo_6]
    ##AUDv
    audv_index = np.arange(238, 250)
    audv = counts[audv_index].values
    audv_1 = (audv[0] + audv[1]) // 2
    audv_2_3 = (audv[2] + audv[3]) // 2
    audv_4 = (audv[4] + audv[5]) // 2
    audv_5 = (audv[6] + audv[7]) // 2
    audv_6 = int(np.mean(audv[8:]))
    audv_counts = [audv_1, audv_2_3, audv_4, audv_5, audv_6]
    ## VIsal
    visal_index = np.arange(254, 266)
    visal = counts[visal_index].values
    visal_1 = (visal[0] + visal[1]) // 2
    visal_2_3 = (visal[2] + visal[3]) // 2
    visal_4 = (visal[4] + visal[5]) // 2
    visal_5 = (visal[6] + visal[7]) // 2
    visal_6 = int(np.mean(visal[8:]))
    visal_counts = [visal_1, visal_2_3, visal_4, visal_5, visal_6]
    ## VISam
    visam_index = np.arange(268, 280)
    visam = counts[visam_index].values
    visam_1 = (visam[0] + visam[1]) // 2
    visam_2_3 = (visam[2] + visam[3]) // 2
    visam_4 = (visam[4] + visam[5]) // 2
    visam_5 = (visam[6] + visam[7]) // 2
    visam_6 = int(np.mean(visam[8:]))
    visam_counts = [visam_1, visam_2_3, visam_4, visam_5, visam_6]
    ##VISI
    visi_index = np.arange(282, 294)
    visi = counts[visi_index].values
    visi_1 = (visi[0] + visi[1]) // 2
    visi_2_3 = (visi[2] + visi[3]) // 2
    visi_4 = (visi[4] + visi[5]) // 2
    visi_5 = (visi[6] + visi[7]) // 2
    visi_6 = int(np.mean(visi[8:]))
    visi_counts = [visi_1, visi_2_3, visi_4, visi_5, visi_6]
    ##VISp
    visp_index = np.arange(296, 308)
    visp = counts[visp_index].values
    visp_1 = (visp[0] + visp[1]) // 2
    visp_2_3 = (visp[2] + visp[3]) // 2
    visp_4 = (visp[4] + visp[5]) // 2
    visp_5 = (visp[6] + visp[7]) // 2
    visp_6 = int(np.mean(visp[8:])) 
    visp_counts = [visp_1, visp_2_3, visp_4, visp_5, visp_6]
    ##VISpl
    vispl_index = np.arange(310, 322)
    vispl = counts[vispl_index].values
    vispl_1 = (vispl[0] + vispl[1]) // 2
    vispl_2_3 = (vispl[2] + vispl[3]) // 2
    vispl_4 = (vispl[4] + vispl[5]) // 2
    vispl_5 = (vispl[6] + vispl[7]) // 2
    vispl_6 = int(np.mean(vispl[8:]))
    vispl_counts = [vispl_1, vispl_2_3, vispl_4, vispl_5, vispl_6]
    ##VISpm
    vispm_index = np.arange(324, 336)
    vispm = counts[vispm_index].values
    vispm_1 = (vispm[0] + vispm[1]) // 2
    vispm_2_3 = (vispm[2] + vispm[3]) // 2
    vispm_4 = (vispm[4] + vispm[5]) // 2
    vispm_5 = (vispm[6] + vispm[7]) // 2
    vispm_6 = int(np.mean(vispm[8:]))
    vispm_counts = [vispm_1, vispm_2_3, vispm_4, vispm_5, vispm_6]
    ##VISli
    visli_index = np.arange(338, 350)
    visli = counts[visli_index].values
    visli_1 = (visli[0] + visli[1]) // 2
    visli_2_3 = (visli[2] + visli[3]) // 2
    visli_4 = (visli[4] + visli[5]) // 2
    visli_5 = (visli[6] + visli[7]) // 2
    visli_6 = int(np.mean(visli[8:]))
    visli_counts = [visli_1, visli_2_3, visli_4, visli_5, visli_6]
    ##VISpor
    vispor_index = np.arange(352, 364)
    vispor = counts[vispor_index].values
    vispor_1 = (vispor[0] + vispor[1]) // 2
    vispor_2_3 = (vispor[2] + vispor[3]) // 2
    vispor_4 = (vispor[4] + vispor[5]) // 2
    vispor_5 = (vispor[6] + vispor[7]) // 2
    vispor_6 = int(np.mean(vispor[8:]))
    vispor_counts = [vispor_1, vispor_2_3, vispor_4, vispor_5, vispor_6]
    ##RSPagl
    rspagl_index = np.arange(494, 504)
    rspagl = counts[rspagl_index].values
    rspagl_1 = (rspagl[0] + rspagl[1]) // 2
    rspagl_2_3 = (rspagl[2] + rspagl[3]) // 2
    rspagl_4 = 0
    rspagl_5 = (rspagl[4] + rspagl[5]) // 2
    rspagl_6 = int(np.mean(rspagl[6:]))
    rspagl_counts = [rspagl_1, rspagl_2_3, rspagl_4, rspagl_5, rspagl_6]
    ##RSPd
    rspd_index = np.arange(506, 516)
    rspd = counts[rspd_index].values
    rspd_1 = (rspd[0] + rspd[1]) // 2
    rspd_2_3 = (rspd[2] + rspd[3]) // 2
    rspd_4 = 0
    rspd_5 = (rspd[4] + rspd[5]) // 2
    rspd_6 = int(np.mean(rspd[6:]))
    rspd_counts = [rspd_1, rspd_2_3, rspd_4, rspd_5, rspd_6]
    ##RSPv
    rspv_index = np.arange(518, 528)
    rspv = counts[rspv_index].values
    rspv_1 = (rspv[0] + rspv[1]) // 2
    rspv_2_3 = (rspv[2] + rspv[3]) // 2
    rspv_4 = 0
    rspv_5 = (rspv[4] + rspv[5]) // 2
    rspv_6 = int(np.mean(rspv[6:]))
    rspv_counts = [rspv_1, rspv_2_3, rspv_4, rspv_5, rspv_6]
    ##VISa
    visa_index = np.arange(532, 544)
    visa = counts[visa_index].values
    visa_1 = (visa[0] + visa[1]) // 2
    visa_2_3 = (visa[2] + visa[3]) // 2
    visa_4 = (visa[4] + visa[5]) // 2
    visa_5 = (visa[6] + visa[7]) // 2
    visa_6 = int(np.mean(visa[8:]))
    visa_counts = [visa_1, visa_2_3, visa_4, visa_5, visa_6]
    ##VISrl
    visrl_index = np.arange(546, 558)
    visrl = counts[visrl_index].values
    visrl_1 = (visrl[0] + visrl[1]) // 2
    visrl_2_3 = (visrl[2] + visrl[3]) // 2
    visrl_4 = (visrl[4] + visrl[5]) // 2
    visrl_5 = (visrl[6] + visrl[7]) // 2
    visrl_6 = int(np.mean(visrl[8:]))
    visrl_counts = [visrl_1, visrl_2_3, visrl_4, visrl_5, visrl_6]
    ##ORBI
    orbi_index = np.arange(418, 428)
    orbi = counts[orbi_index].values
    orbi_1 = (orbi[0] + orbi[1]) // 2
    orbi_2_3 = (orbi[2] + orbi[3]) // 2
    orbi_4 = 0
    orbi_5 = (orbi[4] + orbi[5]) // 2
    orbi_6 =int(np.mean(orbi[6:]))
    orbi_counts = [orbi_1, orbi_2_3, orbi_4, orbi_5, orbi_6]
    ##ORBm
    orbm_index = np.arange(430, 440)
    orbm = counts[orbm_index].values
    orbm_1 = (orbm[0] + orbm[1]) // 2
    orbm_2_3 = (orbm[2] + orbm[3]) // 2
    orbm_4 = 0
    orbm_5 = (orbm[4] + orbm[5]) // 2
    orbm_6 = int(np.mean(orbm[6:]))
    orbm_counts = [orbm_1, orbm_2_3, orbm_4, orbm_5, orbm_6]
    ##ORBvl
    orbvl_index = np.arange(442, 452)
    orbvl = counts[orbvl_index].values
    orbvl_1 = (orbvl[0] + orbvl[1]) // 2
    orbvl_2_3 = (orbvl[2] + orbvl[3]) // 2
    orbvl_4 = 0
    orbvl_5 = (orbvl[4] + orbvl[5]) // 2
    orbvl_6 = int(np.mean(orbvl[6:]))
    orbvl_counts = [orbvl_1, orbvl_2_3, orbvl_4, orbvl_5, orbvl_6]
    ##ACAd
    acad_index = np.arange(368, 378)
    acad = counts[acad_index].values
    acad_1 = (acad[0] + acad[1]) // 2
    acad_2_3 = (acad[2] + acad[3]) // 2
    acad_4 = 0
    acad_5 = (acad[4] + acad[5]) // 2
    acad_6 = int(np.mean(acad[6:]))
    acad_counts = [acad_1, acad_2_3, acad_4, acad_5, acad_6]
    ##ACAv
    acav_index = np.arange(380, 390)
    acav = counts[acav_index].values
    acav_1 = (acav[0] + acav[1]) // 2
    acav_2_3 = (acav[2] + acav[3]) // 2
    acav_4 = 0
    acav_5 = (acav[4] + acav[5]) // 2
    acav_6 = int(np.mean(acav[6:]))
    acav_counts = [acav_1, acav_2_3, acav_4, acav_5, acav_6]
    ##PL
    pl_index = np.arange(392, 402)
    pl = counts[pl_index].values
    pl_1 = (pl[0] + pl[1]) // 2
    pl_2_3 = (pl[2] + pl[3]) // 2
    pl_4 = 0
    pl_5 = (pl[4] + pl[5]) // 2
    pl_6 = int(np.mean(pl[6:]))
    pl_counts = [pl_1, pl_2_3, pl_4, pl_5, pl_6]
    ##ILA
    ila_index = np.arange(404, 414)
    ila = counts[ila_index].values
    ila_1 = (ila[0] + ila[1]) // 2
    ila_2_3 = (ila[2] + ila[3]) // 2
    ila_4 = 0
    ila_5 = (ila[4] + ila[5]) // 2
    ila_6 = int(np.mean(ila[6:]))
    ila_counts = [ila_1, ila_2_3, ila_4, ila_5, ila_6]
    ##VISC
    visc_index = np.arange(180, 192)
    visc = counts[visc_index].values
    visc_1 = (visc[0] + visc[1]) // 2
    visc_2_3 = (visc[2] + visc[3]) // 2
    visc_4 = (visc[4] + visc[5]) // 2
    visc_5 = (visc[6] + visc[7]) // 2
    visc_6 = int(np.mean(visc[8:]))
    visc_counts = [visc_1, visc_2_3, visc_4, visc_5, visc_6]
    ##GU
    gu_index = np.arange(166, 178)
    gu = counts[gu_index].values
    gu_1 = (gu[0] + gu[1]) // 2
    gu_2_3 = (gu[2] + gu[3]) // 2
    gu_4 = (gu[4] + gu[5]) // 2
    gu_5 = (gu[6] + gu[7]) // 2
    gu_6 = int(np.mean(gu[8:]))
    gu_counts = [gu_1, gu_2_3, gu_4, gu_5, gu_6]
    #Ald
    ald_index = np.arange(456, 466)
    ald = counts[ald_index].values
    ald_1 = (ald[0] + ald[1]) // 2
    ald_2_3 = (ald[2] + ald[3]) // 2
    ald_4 = 0
    ald_5 = (ald[4] + ald[5]) // 2
    ald_6 = int(np.mean(ald[6:]))
    ald_counts = [ald_1, ald_2_3, ald_4, ald_5, ald_6]
    ##ALp
    alp_index = np.arange(468, 478)
    alp = counts[alp_index].values
    alp_1 = (alp[0] + alp[1]) // 2
    alp_2_3 = (alp[2] + alp[3]) // 2
    alp_4 = 0
    alp_5 = (alp[4] + alp[5]) // 2
    alp_6 = int(np.mean(alp[6:]))
    alp_counts = [alp_1, alp_2_3, alp_4, alp_5, alp_6]
    ##ALv
    alv_index = np.arange(480, 490)
    alv = counts[alv_index].values
    alv_1 = (alv[0] + alv[1]) // 2
    alv_2_3 = (alv[2] + alv[3]) // 2
    alv_4 = 0
    alv_5 = (alv[4] + alv[5]) // 2
    alv_6 = int(np.mean(alv[6:]))
    alv_counts = [alv_1, alv_2_3, alv_4, alv_5, alv_6]
    ##PERI
    peri_index = np.arange(574, 584)
    peri = counts[peri_index].values
    peri_1 = (peri[0] + peri[1]) // 2
    peri_2_3 = (peri[2] + peri[3]) // 2
    peri_4 = 0
    peri_5 = (peri[4] + peri[5]) // 2
    peri_6 = int(np.mean(peri[6:])) 
    peri_counts = [peri_1, peri_2_3, peri_4, peri_5, peri_6]
    #TEa
    tea_index = np.arange(560, 572)
    tea = counts[tea_index].values
    tea_1 = (tea[0] + tea[1]) // 2
    tea_2_3 = (tea[2] + tea[3]) // 2
    tea_4 = (tea[4] + tea[5]) // 2
    tea_5 = (tea[6] + tea[7]) // 2
    tea_6 = int(np.mean(tea[8:]))
    tea_counts = [tea_1, tea_2_3, tea_4, tea_5, tea_6]
    ##ECT
    ect_index = np.arange(586, 596)
    ect = counts[ect_index].values
    ect_1 = (ect[0] + ect[1]) // 2
    ect_2_3 = (ect[2] + ect[3]) // 2
    ect_4 = 0
    ect_5 = (ect[4] + ect[5]) // 2
    ect_6 = int(np.mean(ect[6:]))
    ect_counts = [ect_1, ect_2_3, ect_4, ect_5, ect_6]

    region_name = ['FRP', 'MOp', 'MOs', 'SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr', 'SSp-un', 'SSs', 'AUDd', 'AUDp', 'AUDpo', 'AUDv', 'VISal', 'VISam',
                    'VISl', 'VISp','VISpl',  'VISpm', 'VISli', 'VISpor', 'RSPagl', 'RSPd', 'RSPv', 'VISa', 'VISrl', 'ORBl', 'ORBm', 'ORBvl', 'ACAd', 'ACAv', 'PL', 'ILA', 'VISC',
                    'GU', 'AId', 'AIp', 'AIv', 'PERl', 'TEa', 'ECT']# num = 43
    counts_list = [frp_counts, mop_counts, mos_counts, ssp_n_counts, ssp_bfd_counts, ssp_ll_counts, ssp_m_counts, ssp_ul_counts, ssp_tr_counts, ssp_un_counts, sss_count, 
                    audd_counts,audp_counts, audpo_counts, audv_counts, visal_counts, visam_counts, visi_counts, visp_counts, vispl_counts, vispm_counts, visli_counts, vispor_counts, 
                    rspagl_counts, rspd_counts, rspv_counts, visa_counts, visrl_counts, orbi_counts, orbm_counts, orbvl_counts, acad_counts, acav_counts, pl_counts, ila_counts, 
                    visc_counts, gu_counts, ald_counts, alp_counts, alv_counts, peri_counts, tea_counts, ect_counts]
    region_counts = dict(zip(region_name, counts_list))
    csv_root = os.path.join(csv_path, '..')
    write_csv_path = os.path.join(csv_root, 'cortex_cells_counts.csv')
    firts_line = ['Region', 'layer 1', 'layer 2/3', 'layer 4', 'layer 5', 'layer 6','SUM']
    with open(write_csv_path, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(firts_line)
        for i in range(len(region_name)):
            write_line = [region_name[i], region_counts[region_name[i]][0], region_counts[region_name[i]][1], region_counts[region_name[i]][2], region_counts[region_name[i]][3],
                            region_counts[region_name[i]][4], sum(region_counts[region_name[i]])]
            csv_write.writerow(write_line)

if __name__ == "__main__":
    root = Data_root
    data_name = Stats['group_data_name']
    for k in range(len(data_name)):
        csv_path = os.path.join(root, data_name[k], 'whole_brain_cell_counts')
        cortex_stats(csv_path)
        print('finished {}'.format(data_name[k]))
