function [Cn1_inv, vn, on] = blockm_inv(Cn_inv, kn1, k_n1n1)
    kn1_T = kn1';
    vn = k_n1n1 - kn1 * Cn_inv * kn1_T;
    ivn = inv(vn);
    on = (Cn_inv * kn1_T)*(-1 * ivn) ;
    Cn1_inv_top_left = Cn_inv - on * (kn1 * Cn_inv);
    Cn1_inv = [Cn1_inv_top_left, on;
               on', ivn];
end
