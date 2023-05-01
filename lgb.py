import datetime
import lightgbm as lgb
import numpy as np
hour = 3600


def get_base_lgb_params(cpu_time, num_rows, num_cols):
    tratio = (cpu_time / hour)
    n_estimators = int(max(1, min(300, 100 * tratio ** 0.5)))
 
    lgb_params = {
        'n_estimators': n_estimators,
        'learning_rate': min(0.5, 0.3 * (n_estimators / 100) ** -0.4),
        'colsample_bytree': max(0.05, min(0.4, 
                                (5000 / num_cols) ** 0.7 * tratio ** 0.5
                                         )),
        'colsample_bynode': 0.8,
        'subsample': min(0.8, (50000 / num_rows) ** 0.8),
        'subsample_freq': 1,
        'reg_alpha': 1e-4,
        'reg_lambda': 1e-2,
        'subsample_for_bin': int(25000 * tratio ** 0.3),
        }

    return lgb_params

def rescale_lgb_params(lgb_params, ratio):
    print('  rescaling to {:.1f}x\n'.format(ratio))

    new_ss = min(0.9, lgb_params['subsample'] * ratio ** 0.7)
    new_csample = min(0.4, lgb_params['colsample_bytree'] 
                                  * ratio ** 0.7)
    new_n_est = min(1000, lgb_params['n_estimators'] * ratio ** 1.0
            * (lgb_params['subsample'] / new_ss) ** 0.5
            * (lgb_params['colsample_bytree'] / new_csample) ** 0.7 )
    new_lr_mult = (new_n_est / lgb_params['n_estimators']) ** -0.4

    lgb_params['subsample'] = new_ss
    lgb_params['colsample_bytree'] = new_csample

    lgb_params['n_estimators'] = int(new_n_est)
    lgb_params['learning_rate'] *= new_lr_mult

    lgb_params['subsample_for_bin'] = max(10000, min(100000, int(
                lgb_params['subsample_for_bin'] 
                                     * ratio ** 0.4)))
    return lgb_params

def get_final_lgb_params(lgb_params, ratio):
    new_ss = min(0.9, lgb_params['subsample'] * ratio ** 0.5)
    new_csample = min(0.5, lgb_params['colsample_bytree'] 
                                  * ratio ** 0.5)
    new_n_est = min(2000, 
                    lgb_params['n_estimators'] * ratio ** 1.0
                * (lgb_params['subsample'] / new_ss) ** 0.8
                * (lgb_params['colsample_bytree'] / new_csample) ** 0.8 )
    new_lr_mult = (new_n_est / lgb_params['n_estimators']) ** -0.4

    lgb_params['subsample'] = new_ss
    lgb_params['colsample_bytree'] = new_csample

    lgb_params['n_estimators'] = int(new_n_est)
    lgb_params['learning_rate'] *= new_lr_mult

    lgb_params['subsample_for_bin'] = max(20000, min(100000, int(
                lgb_params['subsample_for_bin'] 
                                     * ratio ** 0.5)))
    return lgb_params

def get_full_lgb_params(lgb_params, ratio):
    new_n_est = min(20000, 
                    lgb_params['n_estimators'] * ratio ** 0.7)
    new_n_leaves = lgb_params['num_leaves'] * ratio ** 0.3
    new_lr_mult = (new_n_est / lgb_params['n_estimators']) ** -0.4

    lgb_params['n_estimators'] = int(new_n_est)
    lgb_params['num_leaves'] = int(new_n_leaves)
    lgb_params['learning_rate'] *= new_lr_mult
    lgb_params['subsample_for_bin'] = max(20000, min(200000, int(
                    lgb_params['subsample_for_bin'] 
                                         * ratio ** 0.7)))
    return lgb_params



def get_lgb_param_dict(lgb_params, ratio):
    new_ss = min(0.9, lgb_params['subsample'] * ratio ** 0.7)
    new_csample = min(0.4, lgb_params['colsample_bytree'] 
                                      * ratio ** 0.5)
    new_n_est = min(7000, lgb_params['n_estimators'] * ratio ** 1.0
            * (lgb_params['subsample'] / new_ss) ** 0.8
            * (lgb_params['colsample_bytree'] / new_csample) ** 0.8 )
    new_lr_mult = (new_n_est / lgb_params['n_estimators']) ** -0.4
    lgb_param_dict = {
        'n_estimators': [int(new_n_est)],
        'learning_rate': (10 ** np.arange(-1.3, -0.3, 0.1)).round(3)
                             * (int(new_n_est) / 300) ** -0.4,
        'colsample_bytree': np.arange(0.1, 0.7, 0.05).round(2)
                             if new_csample > 0.3
                            else (new_csample
                                    * np.exp(np.arange(-0.5, 0.5, 0.1)) ),
        'subsample': np.arange(0.8, 0.9, 0.01).round(2)
                             if new_ss > 0.4
                        else (new_ss * np.exp(np.arange(-0.5, 0.5, 0.1)) ),

        'subsample_for_bin': [max(10000, min(100000, int(
                lgb_params['subsample_for_bin'] * ratio ** 0.8)))],
        

        'colsample_bynode': np.arange(0.7, 0.91, 0.1).round(2),
        'num_leaves': [31, 31, 31, 50, 50, 70,] if new_n_est > 100
                        else [20, 31],
        'max_depth': [9, 10, -1, -1, -1,],
        'min_child_weight': (10 ** np.arange(-4, -1.5, 0.1)).round(4),
        'min_child_samples': (10 ** np.arange(1.0, 2.0, 0.05)).astype(int),
        'reg_alpha': (10 ** np.arange(-5, -3, 0.1)).round(6),
        'reg_lambda': (10 ** np.arange(-3, 1.0, 0.2)).round(6),
        'subsample_freq': [1],
    }
    return lgb_param_dict


def get_lgb_model(params, task_type, output_size, final_metric):
    lgb_params = {'seed': datetime.datetime.now().microsecond}
    lgb_params.update(params)
    if task_type == 'single-label':
        if output_size > 2: # multi-class
            lgb_params.update(
                {'num_class': output_size,})
        else: # binary
            pass;
        model = lgb.LGBMClassifier(**lgb_params,)

    elif task_type == 'multi-label':
        model = lgb.LGBMClassifier(**lgb_params)

    elif task_type == 'continuous':
        model = lgb.LGBMRegressor(**lgb_params)
    else: 
        raise NotImplementedError
    return model

