import torch
from utils import plotsAnalysis
import os
import shutil
if __name__ == '__main__':
    #pathnamelist = ['/home/sr365/MM_Bench/Tandem/models/Yang/sweep3/noconv/',
    #               '/home/sr365/MM_Bench/Tandem/models/Yang/sweep3/conv_444_334_112/',
    #                ]
    pathnamelist = ['/home/sr365/MM_Bench/NA/models/Omar']
                #'/home/sr365/MM_Bench/NA/models/Peurifoy_big']
                #'/home/sr365/MM_Bench/cINN/models/Peurifoy_rand/']
                #
                #'/home/sr365/MM_Bench/VAE/models/Peurifoy_4th']
                #'/home/sr365/MM_Bench/VAE/models/Peurifoy_3rd']
                
                #'/home/sr365/MM_Bench/INN_FrEIA/models/Peurifoy']# #'/home/sr365/MM_Bench/MDN/models/Peurifoy']
                    #'/home/sr365/MM_Bench/NA/models/Chen/sweep5/lr0.0001/reg0/']
    #pathnamelist= ['/home/sr365/MM_Bench/NA/models/Yang_sim/conv_444_435_211/']
    pathnamelist = ['../Forward/models/filtered']
    pathfrom = ['../Forward/models/Peurifoy']

    filtered = []
    for pathname in pathfrom:
        folders = os.listdir(pathname)
        for f in folders:
            all = f.split('_')
            el = tuple(map(float,all[:5]))
            # 0 = # layers
            # 1 = # nodes
            # 2 = l_rate
            # 3 = reg scale
            # 4 = l_rate decay
            # 5 = count

            #layers-> 8,9,11,12;
            #if el[0]>=10 and el[1]>=1500:
            # FORWARD PEURIFOY: Nodes 1500 - 2000, layers = 7 - 15, lr_decay_rate = 0.1-0.2,
            if (el[0] >= 10 and el[0] <= 15) and (el[1] >= 1500 and el[1] <= 2000) and el[3] in (1e-3,1e-4,0,1e-5) and el[2] in (0.1,0.01,1e-3,1e-4) and el[4] in [0.1,0.2,0.3]:
                filtered.append(f)

        for f in filtered:
            shutil.copytree(os.path.join(pathname,f),os.path.join(pathnamelist[0],f))

    for pathname in pathnamelist:
        
        # Forward: Convolutional swipe
        #plotsAnalysis.HeatMapBVL('kernel_first','kernel_second','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir='models/'+pathname,feature_1_name='kernel_first',feature_2_name='kernel_second')
        

        # General: lr vs layernum
        plotsAnalysis.HeatMapBVL('num_layers','reg_scale','layer vs unit Heat Map',save_name=pathname + 'layer vs reg_scale.png',
                                HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='reg_scale')

        # General: Complexity swipe
<<<<<<< HEAD
        plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name=pathname + 'layer vs unit_heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='linear_unit')
=======
        plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name=pathname + 'Peurifoy_node_vs_layers_heatmap.png',
                                 HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='linear_unit')

        shutil.rmtree(pathname)
>>>>>>> 59e71ae73cce8b748bde634d82480d2d2ba2af13
        
        # General: lr vs layernum
        plotsAnalysis.HeatMapBVL('num_layers','lr','layer vs unit Heat Map',save_name=pathname + 'layer vs lr_heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='lr')

        # MDN: num layer and num_gaussian
        # plotsAnalysis.HeatMapBVL('num_layers','num_gaussian','layer vs num_gaussian Heat Map',save_name=pathname + 'layer vs num_gaussian heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='num_gaussian')
        
        # General: Reg scale and num_layers
        #plotsAnalysis.HeatMapBVL('num_layers','reg_scale','layer vs reg Heat Map',save_name=pathname + 'layer vs reg_heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='reg_scale')
        
        # # VAE: kl_coeff and num_layers
        # plotsAnalysis.HeatMapBVL('num_layers','kl_coeff','layer vs kl_coeff Heat Map',save_name=pathname + 'layer vs kl_coeff_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='linear_d',feature_2_name='kl_coeff')

        # # VAE: kl_coeff and dim_z
        # plotsAnalysis.HeatMapBVL('dim_z','kl_coeff','kl_coeff vs dim_z Heat Map',save_name=pathname + 'kl_coeff vs dim_z Heat Map heatmap.png',
        #                      HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='kl_coeff')

        # # VAE: dim_z and num_layers
        # plotsAnalysis.HeatMapBVL('dim_z','num_layers','layer vs unit Heat Map',save_name=pathname + 'layer vs dim_z Heat Map heatmap.png',
        #                      HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_d')
        
        # # VAE: dim_z and num_unit
        # plotsAnalysis.HeatMapBVL('dim_z','num_unit','dim_z vs unit Heat Map',save_name=pathname + 'dim_z vs unit Heat Map heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_unit')

        # # General: Reg scale and num_unit (in linear layer)
        # plotsAnalysis.HeatMapBVL('reg_scale','num_unit','reg_scale vs unit Heat Map',save_name=pathname + 'reg_scale vs unit_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='reg_scale',feature_2_name='linear_unit')
        
        # # cINN or INN: Couple layer num and lambda mse
        # plotsAnalysis.HeatMapBVL('couple_layer_num','lambda_mse','couple_num vs lambda mse Heat Map',save_name=pathname + 'couple_num vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='lambda_mse')
        
        # # cINN or INN: lambda z and lambda mse
        # plotsAnalysis.HeatMapBVL('lambda_z','lambda_mse','lambda_z vs lambda mse Heat Map',save_name=pathname + 'lambda_z vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='lambda_z',feature_2_name='lambda_mse')

        # # cINN or INN: lambda rev and lambda mse
        # plotsAnalysis.HeatMapBVL('lambda_rev','lambda_mse','lambda_rev vs lambda mse Heat Map',save_name=pathname + 'lambda_rev vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='lambda_rev',feature_2_name='lambda_mse')

        # # cINN or INN: lzeros_noise_scale and lambda mse
        # plotsAnalysis.HeatMapBVL('zeros_noise_scale','lambda_mse','zeros_noise_scale vs lambda mse Heat Map',save_name=pathname + 'zeros_noise_scale vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='zeros_noise_scale',feature_2_name='lambda_mse')

        # # cINN or INN: zeros_noise_scaleand y_noise_scale
        # plotsAnalysis.HeatMapBVL('zeros_noise_scale','y_noise_scale','zeros_noise_scale vs y_noise_scale Heat Map',save_name=pathname + 'zeros_noise_scale vs y_noise_scale_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='zeros_noise_scale',feature_2_name='y_noise_scale')

        

        # # cINN or INN: Couple layer num and reg scale
        # plotsAnalysis.HeatMapBVL('couple_layer_num','reg_scale','layer vs unit Heat Map',save_name=pathname + 'couple_layer_num vs reg_scale_heatmap.png',
        #                          HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='reg_scale')
        
        
        # # INN: Couple layer num and dim_pad
        # plotsAnalysis.HeatMapBVL('couple_layer_num','dim_tot','couple_layer_num vs dim pad Heat Map',save_name=pathname + 'couple_layer_num vs dim pad _heatmap.png',
        #                         HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='dim_tot')

        # # INN: Lambda_mse num and dim_pad
        # plotsAnalysis.HeatMapBVL('lambda_mse','dim_tot','lambda_mse vs dim_tot Heat Map',save_name=pathname + 'lambda_mse vs dim_tot_heatmap.png',
        #                         HeatMap_dir=pathname, feature_1_name='lambda_mse',feature_2_name='dim_tot')
        
        # # INN: Couple layer num and dim_z
        # plotsAnalysis.HeatMapBVL('couple_layer_num','dim_z','couple_layer_num vs dim_z Heat Map',save_name=pathname + 'couple_layer_num vs dim_z_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='dim_z')
