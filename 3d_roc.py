import numpy as np

def AUC_calculate(target, predicted, adaptive_topcut = False, seg_mode=1, normalize=False):
    """
    C. -I. Chang, "An Effective Evaluation Tool for Hyperspectral Target Detection: 3D Receiver Operating Characteristic Curve Analysis,"
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 6, pp. 5131-5153, June 2021.
    Z. Li, Y. Wang, C. Xiao, Q. Ling, Z. Lin and W. An, "You Only Train Once: Learning a General Anomaly Enhancement
    Network with Random Masks for Hyperspectral Anomaly Detection," in IEEE Transactions on Geoscience and Remote Sensing,
     doi: 10.1109/TGRS.2023.3258067.
    Input:
        target.shape = [n,1]
        predicted.shape = [n,1]
        adaptive_topcut: if True, the function will return the adaptive version of 3D ROC metrics in Li'paper ,the predicted
                        map will be truncated on the median value of predicted scores on the target GT.
        seg_mode: if seg_mode==1, the equation (7) in  Chang's paper is used; else the equation (9) in Chang's paper is used.
        normalize: if True, the outputs are normalize auc in  Chang's paper.
    Output:
        tau : List,segmentation thresholds.
        PD: List.
        PF: List.
        PD_PF_auc:[0,1] The higher the PD_PF_auc, the better  the detector.
        PF_tau_auc:[0,1] The smaller the PF_tau_auc, the better the detector.
        PD_tau_auc: [0,1] The higher the PD_tau_auc, the better  the detector.
        SNPR（dB）: [-inf,inf] The higher the SNPR, the betters the detector.
    """
    if np.sum(np.abs(predicted-np.mean(predicted))) == 0:
        return [],[],[], 0.5, 0, 1, 0
    else:
        # predicted[predicted==np.max(predicted)] = 1*np.max(predicted)
        if adaptive_topcut:
            max_limit = np.median(predicted[target > 0])
            predicted[predicted> max_limit ] = max_limit
            if np.sum(np.abs(predicted - np.mean(predicted))) == 0:
                return [],[],[], 0.5, 0, 1, 0
            # predicted[(predicted > max_limit) &  (target > 0)] = max_limit
        target = ((target - target.min()) /
                  (target.max() - target.min()))
        predicted = ((predicted-predicted.min()) /
                                (predicted.max()-predicted.min()))
        anomaly_map = target
        normal_map = 1-target
        taus = np.unique(predicted)
        num = taus.size
        PF = np.zeros([num,1])
        PD = np.zeros([num,1])
        for index in range(num):
            tau = taus[index]
            if seg_mode ==1:
                anomaly_map_1 = np.double(predicted>=tau)
            else:
                anomaly_map_1 = np.double(predicted > tau)
            PF[index] = np.sum(anomaly_map_1*normal_map)/np.sum(normal_map)
            PD[index] = np.sum(anomaly_map_1*anomaly_map)/np.sum(anomaly_map)
        PD_PF_auc = np.sum((PF[0:num-1,:]-PF[1:num,:])*(PD[1:num]+PD[0:num-1])/2)
        PF_tau_auc = np.trapz(PF.squeeze(),taus.squeeze())
        PD_tau_auc = np.trapz(PD.squeeze(), taus.squeeze())
        SNPR = 10*np.log10(PD_tau_auc/PF_tau_auc)
        if normalize:
            a1 = PD[-1]
            a0 = PD[0]
            b1 = PF[-1]
            b0 = PF[0]
            PD_PF_auc = (PD_PF_auc - a1) / ((a0 - a1) * (b0 - b1))
            PD_tau_auc = (PD_tau_auc - a1) / (a0 - a1)
            PF_tau_auc = (PF_tau_auc - b1) / (b0 - b1)
            SNPR = 10*np.log10(PD_tau_auc/PF_tau_auc)
        return list(taus), list(PD), list(PF), PD_PF_auc, PF_tau_auc, PD_tau_auc, SNPR



if __name__ == '__main__':
    gt = np.zeros([20,20])
    gt[:5,:5] =1
    detect1 = np.zeros_like(gt)
    detect1[:5, :3] = 10
    detect2 = np.zeros_like(gt)
    detect2[:5, :3] = 10
    detect2[10, 10] = 100

    print('3D ROC metrics in Chang\'s paper. Eg1 should be better than eg2.')
    taus, PDs, PFs, PD_PF_auc, PF_tau_auc, PD_tau_auc, AUC_SNPR = AUC_calculate(gt.reshape(-1), detect1.reshape(-1),
                                                                adaptive_topcut=False)
    print(
        'Eg1 AUC_pdpf: {:.3f} AUC_pf_t: {:.3f} AUC_pd_t: {:.3f} SNPR: {:.3f}'.format(PD_PF_auc, PF_tau_auc, PD_tau_auc,
                                                                                     AUC_SNPR))

    taus, PDs, PFs, PD_PF_auc, PF_tau_auc, PD_tau_auc, AUC_SNPR = AUC_calculate(gt.reshape(-1), detect2.reshape(-1),
                                                                adaptive_topcut=False)
    print(
        'Eg2 AUC_pdpf: {:.3f} AUC_pf_t: {:.3f} AUC_pd_t: {:.3f} SNPR: {:.3f}'.format(PD_PF_auc, PF_tau_auc, PD_tau_auc,
                                                                                     AUC_SNPR))
    print('3D ROC metrics in Li\'s paper.')
    taus, PDs, PFs, PD_PF_auc, PF_tau_auc, PD_tau_auc, AUC_SNPR = AUC_calculate(gt.reshape(-1), detect1.reshape(-1),
                                                                adaptive_topcut=True)
    print('Eg1 AUC_pdpf: {:.3f} AAUC_pf_t: {:.3f} AAUC_pd_t: {:.3f} ASNPR: {:.3f}'.format(PD_PF_auc, PF_tau_auc,
                                                                                          PD_tau_auc, AUC_SNPR))

    taus, PDs, PFs, PD_PF_auc, PF_tau_auc, PD_tau_auc, AUC_SNPR = AUC_calculate(gt.reshape(-1), detect2.reshape(-1),
                                                                adaptive_topcut=True)
    print('Eg2 AUC_pdpf: {:.3f} AAUC_pf_t: {:.3f} AAUC_pd_t: {:.3f} ASNPR: {:.3f}'.format(PD_PF_auc, PF_tau_auc,
                                                                                          PD_tau_auc, AUC_SNPR))

