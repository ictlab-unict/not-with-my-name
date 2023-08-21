'''
    @Author: Roberto Leotta
    @Version: 1.0
    @Date: 08/16/2023

    env:
    conda activate not-w-my-name-env

    How to run:
    python src/inference.py --dataset-folder resources/small-dataset/ --query-img resources/small-dataset/pablo_picasso/ai_generated/102_0.png --model-ckpt resources/ckpts/siamese_not_w_my_name.ckpt --cuda --results-folder results
'''
########## IMPORTs START ##########

# system imports
import os
import time
import argparse

# nn
import torch
from SiameseNet import SiameseNetworkTask, compute_embedding
import torch.nn.functional as F

# utils
import utils
import pandas as pd
from tqdm import tqdm

###### IMPORTs - END ##############

###
###
###

###### CONSTANTs - START ##########

###### CONSTANTs - END ############

###
###
###

###### FUNCTIONs - START ##########

###### FUNCTIONs - END ############

###
###
###

###### MAIN - START ###############

if __name__ == "__main__":    
    ### 
    # initial settings

    parser = argparse.ArgumentParser(description='Not with my name inference\nby Roberto Leotta')
    parser.add_argument('--show-time', dest='show_time', default=True, action='store_true', 
                        help='show processing time')
    parser.add_argument('--debug', required=False, default=False, action='store_true', 
                        help='flag for development debugging')
    
    parser.add_argument('--dataset-folder', dest='dataset_folder', type=str, required=True, 
                        help='dataset folder path')
    parser.add_argument('--results-folder', dest='results_folder', required=False, default='', type=str, 
                        help='results folder path')
    parser.add_argument('--query-img', dest='query_image', type=str, required=True, 
                        help='query image path')
    parser.add_argument('--distance-th', dest='distance_th', type=float, required=False, default=0.5,
                        help='distance threshold for the query image')
    
    parser.add_argument('--cuda', required=False, default=False, action='store_true',
                        help='use CUDA for inference')
    parser.add_argument('--model-ckpt', dest='model_ckpt', required=True, type=str, 
                        help='siamese model checkpoint')
    args = parser.parse_args()

    # check if input folder exists
    DATASET_FOLDER = os.path.abspath(args.dataset_folder)
    if not os.path.exists(DATASET_FOLDER):
        utils.p_error_n_exit('Dataset folder does not exist: {}'.format(DATASET_FOLDER))

    # check if model checkpoint exists
    MODEL_CKPT = os.path.abspath(args.model_ckpt)
    if not os.path.exists(MODEL_CKPT):
        utils.p_error_n_exit('Model checkpoint does not exist: {}'.format(MODEL_CKPT))

    # check if query image exists
    QUERY_IMAGE = os.path.abspath(args.query_image)
    if not os.path.exists(QUERY_IMAGE):
        utils.p_error_n_exit('Query image does not exist: {}'.format(QUERY_IMAGE))

    # create results folder
    if args.results_folder != '':
        RESULTS_FOLDER = os.path.abspath(args.results_folder)
    else:
        RESULTS_FOLDER = os.path.join(utils.ROOT_DIR, 'results')
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # save and print the used config
    with open(os.path.join(RESULTS_FOLDER, 'config.txt'), 'w') as config_file:
        config_file.write('*** CONFIG ***\n')
        for arg in args.__dict__:
            if args.__dict__[arg] is not None:
                config_file.write(str(arg) + ': ' + str(args.__dict__[arg]) + '\n')

    print('*** CONFIG ***\n', args, '\n')

    if args.show_time:
        print('*** START ***\n')
        start_time = time.time()

    ###
    # init device
    if args.cuda and torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    # print used device
    utils.p_info('Using device: {}'.format(DEVICE))

    # init model
    model = SiameseNetworkTask.load_from_checkpoint(MODEL_CKPT, map_location=DEVICE)
    model.eval()
    if args.debug: utils.p_info('Network:\n{}'.format(model)); utils.p_info('')
    utils.p_info('Siamese network loaded from checkpoint: {}'.format(MODEL_CKPT))

    ###
    # create a dataframe with the original images and the relative embeddings;
    # the dataframe will be used to compute the distance between the query image and the original images.
    utils.p_info('Computing embeddings for original images and generated images...')
    original_imgs_list = []
    generated_imgs_list = []
    for root, dirs, files in os.walk(DATASET_FOLDER):
        if files != [] and 'original_paintings' in root:
            total_file = len(files)
        else: continue
        with tqdm(total=total_file, desc='Processing {} embeddings'.format(os.path.basename(os.path.dirname(root)) + '/' + os.path.basename(root))) as pbar:
            for file in files:
                if utils.allowedFile(file):
                    # retrieve the image path, the class name and the original/generated flag
                    img_path = os.path.join(root, file)
                    class_name = os.path.basename(os.path.dirname(root))
                    original_generated = os.path.basename(root)
                    original_generated = 'original' if original_generated == 'original_paintings' else 'generated'
                    # compute the embedding
                    embedding = compute_embedding(img_path, model.embedding_net, DEVICE)

                    if original_generated == 'original':
                        original_imgs_list.append([img_path, class_name, original_generated, embedding])
                    else:
                        generated_imgs_list.append([img_path, class_name, original_generated, embedding])

                    pbar.update(1)

    # create the dataframe
    df_original_imgs = pd.DataFrame(original_imgs_list, columns=['img_path', 'class', 'original_generated', 'embedding'])
    # free up memory
    del original_imgs_list

    # compute the embeddings for the query image
    utils.p_info('Computing embedding for query image...')
    query_embedding = compute_embedding(QUERY_IMAGE, model.embedding_net, DEVICE)

    # add a new column to the dataframe with the distance between the query image and the original images
    df_original_imgs['query_distance'] = None

    # compute the distance
    utils.p_info('Computing distance between query image and original images...')
    for index, row in df_original_imgs.iterrows():
        orginal_embedding = row['embedding']
        distance = F.pairwise_distance(orginal_embedding, query_embedding, p=2).detach().cpu().numpy()[0]

        if distance <= args.distance_th:
            # save the distance
            df_original_imgs.at[index, 'query_distance'] = distance
        else:
            # remove the row in wich the distance is greater than the threshold
            df_original_imgs.drop(index, inplace=True)

    # sort the dataframe by distance
    df_original_imgs = df_original_imgs.sort_values(by=['query_distance'])
    # remove the embedding column
    df_original_imgs = df_original_imgs.drop(columns=['embedding'])
    # save the dataframe
    utils.p_info('Saving results in {}'.format(os.path.join(RESULTS_FOLDER, 'distances.csv')))
    df_original_imgs.to_csv(os.path.join(RESULTS_FOLDER, 'distances.csv'), index=False)

    ###
    # exit

    if args.show_time: print('\n*** END in {:0.2f} secs ***\n'.format(time.time() - start_time))


###### MAIN - END #################

###
###
###

###### COMMENTs ###################