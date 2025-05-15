import argparse

def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default=r'E:/dataset/impairedvideo360_min/ARaw/J_raw10g_rotate_yuv'
                                            , type=str, help='Path to input videos')#E:/dataset/impairedvideo360_min/ARaw/J_raw10g_rotate_yuv E:/dataset/impariedvideos_min/enhancement/g6_en E:/BaiduNetdiskDownload/CSIQ/all-videos E:/dataset/impariedvideos_min/720p_yuv E:/dataset/impairedvideo240_min/480_min_yuv/ E:/dataset/LIVE_Wild_Compressed_Video/tempr'E:/dataset/impariedvideos_min/enhancement/lq_en'
    parser.add_argument('--score_file_path', default='./dataset/odv240_min/odv_720_entest1_g6.json', type=str, help='Path to input subjective score')#./dataset/odv240_min/odv_720_entest3_g6.json  odv240_min/odv_720_10gerpr_subj_score.json ./dataset/LIVE Wild/livewild_subj_score_yuv.json'./dataset/odv240_min/odv_720_entest1_g6.json'
    parser.add_argument('--load_model', default='./save/model_720Pt.pt_7', type=str, help='Path to load checkpoint')#model_720P.pt_7 model_pseudo.pt_21  ./save/model_en22.pt_3 ./save/model_videoset_v3.pt ./save/model_odv247.pt ./save/csiq_casa.pt ./save/model_720P.pt_7 model_en1.pt_303 ./save/swin_tiny_patch4_window7_224.pth './save/model_erpnum.pt_400'
    parser.add_argument('--save_model', default='./save/model_fr6.pt', type=str, help='Path to save checkpoint')
    parser.add_argument('--log_file_name', default='./log/test1.log', type=str, help='Path to save log')

    parser.add_argument('--channel', default=1, type=int, help='channel number of input data, 1 for Y channel, 3 for YUV')
    parser.add_argument('--size_x', default=112, type=int, help='patch size x of segment')#112
    parser.add_argument('--size_y', default=112, type=int, help='patch size y of segment')
    parser.add_argument('--stride_x', default=80, type=int, help='patch stride x between segments')#80 216
    parser.add_argument('--stride_y', default=80, type=int, help='patch stride y between segments')

    parser.add_argument('--learning_rate', default=0.00003, type=float, help='learning rate') #0.0001 0.0003 | 0.00003
    parser.add_argument('--weight_decay', default=0.0002, type=float, help='L2 regularization') #0.0002 | 0.0002
    parser.add_argument('--epochs', default=500, type=int, help='epochs to train')
    parser.add_argument('--multi_gpu', action='store_true', help='whether to use all GPUs')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_opts()
    print(args)
