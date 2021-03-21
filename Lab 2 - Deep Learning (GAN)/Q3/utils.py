from __future__ import division, print_function
import time
import matplotlib.pyplot as plt
import numpy as np
from math import floor
import os
from scipy.linalg import sqrtm
from scipy.spatial import distance_matrix


def calculate_inception_distance_generated_img(generator, model, latent_dim, num_imgs_each_digit, X_test):
    noise = np.random.normal(0, 1, (num_imgs_each_digit * 10, latent_dim))
    tmp = []
    for i in range(num_imgs_each_digit):
        for digit in range(10):
            tmp.append(digit)
    sampled_labels = np.array(tmp)
    gen_imgs = generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    pred_list = []
    pred_fake = model.predict(gen_imgs.reshape(-1, 28, 28, 1))
    pred_real = model.predict(X_test.reshape(-1, 28, 28, 1))

    # calculate mean and covariance statistics
    mu1, sigma1 = pred_real.mean(axis=0), np.cov(pred_real, rowvar=False)
    mu2, sigma2 = pred_fake.mean(axis=0), np.cov(pred_fake, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2)).real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_inception_score_generated_img(generator, model, latent_dim, num_imgs_each_digit, n_split = 10, eps = 1E-16):
    noise = np.random.normal(0, 1, (num_imgs_each_digit * 10, latent_dim))
    tmp = []
    for i in range(num_imgs_each_digit):
        for digit in range(10):
            tmp.append(digit)
    sampled_labels = np.array(tmp)
    gen_imgs = generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    pred_list = []
    for i in range(gen_imgs.shape[0]):
        pred = model.predict(gen_imgs[i,:,:].reshape(1, 28, 28, 1))
        pred_list.append(pred[0])
        # pred[0] = [1.0000000e+00 3.7553105e-20 1.3926897e-13 1.4102576e-16 1.2462111e-23 1.8829189e-09 6.9483339e-14 8.5432184e-15 1.6225489e-13 8.5705963e-17]
    n_part = floor(gen_imgs.shape[0]/n_split)
    scores = list()
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        p_yx = np.array(pred_list[ix_start:ix_end])
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # p_y.shape = (1,10)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # kl_d.shape =  (10, 10) = (num_img_each_digit, num_img_each_digit)
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # sum_kl_d: [0.00302802 0.00023824 0.00023874 0.00023875 0.00039427 0.00023862 0.00022565 0.00023879 0.0002121  0.0002388 ]
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # avg_kl_d:  0.0005291983
        # undo the log
        is_score = np.exp(avg_kl_d)
        # is_score:  1.0005293
        # store
        scores.append(is_score)
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

def calculate_mmd_generated_img(generator, model, latent_dim, num_imgs_each_digit, X_test):
    noise = np.random.normal(0, 1, (num_imgs_each_digit * 10, latent_dim))
    tmp = []
    for i in range(num_imgs_each_digit):
        for digit in range(10):
            tmp.append(digit)
    sampled_labels = np.array(tmp)
    gen_imgs = generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    pred_list = []
    pred_fake = model.predict(gen_imgs.reshape(-1, 28, 28, 1))
    pred_real = model.predict(X_test.reshape(-1, 28, 28, 1))

    # Distance matrices
    Mxx = distance_matrix(pred_real, pred_real)
    Mxy = distance_matrix(pred_real, pred_fake)
    Myy = distance_matrix(pred_fake, pred_fake)  

    scale = Mxx.mean()
    Mxx = np.exp(-Mxx/(scale*2))
    Mxy = np.exp(-Mxy/(scale*2))
    Myy = np.exp(-Myy/(scale*2))
    a = Mxx.mean()+Myy.mean()-2*Mxy.mean()
    mmd = np.sqrt(max(a, 0))

    return mmd

def calculate_mode_score_generated_img(generator, model, latent_dim, num_imgs_each_digit, X_test, n_split = 10, eps = 1E-16):
    noise = np.random.normal(0, 1, (num_imgs_each_digit * 10, latent_dim))
    tmp = []
    for i in range(num_imgs_each_digit):
        for digit in range(10):
            tmp.append(digit)
    sampled_labels = np.array(tmp)
    gen_imgs = generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    pred_list = []
    pred_fake = model.predict(gen_imgs.reshape(-1, 28, 28, 1))
    pred_real = model.predict(X_test.reshape(-1, 28, 28, 1))

    # calculate mean and covariance statistics
    mu1, sigma1 = pred_real.mean(axis=0), np.cov(pred_real, rowvar=False)
    mu2, sigma2 = pred_fake.mean(axis=0), np.cov(pred_fake, rowvar=False)

    kl_1 = pred_fake * (np.log(mu1 + eps) - np.log(np.expand_dims(mu1, 0) + eps))
    kl_2 = mu1*(np.log(mu1 + eps) - np.log(mu2 + eps))


    avg_kl_1 = kl_1.sum(axis=1).mean()
    sum_kl_2 = kl_2.sum()
    # calculate score
    ms = np.exp(avg_kl_1 - sum_kl_2)
    return ms


def calculate_nn_score_generated_img(generator, model, latent_dim, num_imgs_each_digit, X_test):
    noise = np.random.normal(0, 1, (num_imgs_each_digit * 10, latent_dim))
    tmp = []
    for i in range(num_imgs_each_digit):
        for digit in range(10):
            tmp.append(digit)
    sampled_labels = np.array(tmp)
    gen_imgs = generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    pred_list = []
    pred_fake = model.predict(gen_imgs.reshape(-1, 28, 28, 1))
    pred_real = model.predict(X_test[:1000].reshape(-1, 28, 28, 1))

    # Distance matrices
    Mxx = distance_matrix(pred_real, pred_real)
    Mxy = distance_matrix(pred_real, pred_fake)
    Myy = distance_matrix(pred_fake, pred_fake)  
    n0 = Mxx.shape[0]
    n1 = Myy.shape[1]

    label = np.array([0]*n0 + [1]*n1)
    M1 = np.concatenate([Mxx, Mxy], 1)
    M2 = np.concatenate([np.transpose(Mxy, (0,1)), Myy], 1)
    M = np.concatenate([M1, M2], 0)
    M = np.abs(M)
    M_fin = (M+np.diag(np.inf*np.ones(n0+n1)))
    idx = np.argmax(M_fin)
    val = M_fin[idx]

    count = np.zeros(n0+n1)
    count = count + label[idx]
    pred = np.greater(count, 0.5*np.ones(n0+n1)).astype('float')   
    sc = np.equal(label, pred).astype('float').mean()
    return sc