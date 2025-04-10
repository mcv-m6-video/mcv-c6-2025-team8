# With GMFlow it was too difficult to write our own function,
# so what we did was the following:
#
# 1. We cloned the repository from github and created the conda enviroment for gmflow
# 2. We got used to the main.py and the test runs one can do
# 3. Copied our data into the repository
# 4. Modified the inference_on_dir function in evauate.py
#    so it prints out the flow_result in an .npy file format
# 5. Then we run the following command:
# CUDA_VISIBLE_DEVICES=0 python main.py --inference_dir demo/kitti --output_path output/gmflow-kitti --resume pretrained/gmflow_kitti-285701a8.pth
# 6. We used our python scripts and functions to calculate the metrics
#
# We changed only the inference_on_dir function in the file evaluate.py in the repo

# The following implementation of the function inference_on_dir in the file evaluate.py
# is the one we used to compute our flow object a numpy array with GMFlow:

@torch.no_grad()
def inference_on_dir(model,
                     inference_dir,
                     output_path='output',
                     padding_factor=8,
                     inference_size=None,
                     paired_data=False,  # dir of paired testdata instead of a sequence
                     save_flo_flow=False,  # save as .flo for quantitative evaluation
                     save_npy_flow=True,  # Save flow as .npy file
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     pred_bidir_flow=False,
                     fwd_bwd_consistency_check=False,
                     ):
    model.eval()

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filenames = sorted(glob(inference_dir + '/*'))
    print('%d images found' % len(filenames))

    stride = 2 if paired_data else 1

    if paired_data:
        assert len(filenames) % 2 == 0

    for test_id in range(0, len(filenames) - 1, stride):
        image1 = frame_utils.read_gen(filenames[test_id])
        image2 = frame_utils.read_gen(filenames[test_id + 1])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image, for example, HD1K
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        if inference_size is None:
            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].to('cpu'), image2[None].to('cpu'))
        else:
            image1, image2 = image1[None].cuda(), image2[None].cuda()

        # Resize before inference
        if inference_size is not None:
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)

        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             pred_bidir_flow=pred_bidir_flow,
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # Resize back
        if inference_size is not None:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if inference_size is None:
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        else:
            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        # Save flow as .npy file
        if save_npy_flow:
            npy_output_file = os.path.join(output_path,  'gmflow_00045.npy')
            np.save(npy_output_file, flow)
            print(f'Saved flow to {npy_output_file}')

        # Save visualized flow
        output_file = os.path.join(output_path, 'gmflow_00045.png')
        save_vis_flow_tofile(flow, output_file)

        # Also predict backward flow
        if pred_bidir_flow:
            assert flow_pr.size(0) == 2  # [2, H, W, 2]

            if inference_size is None:
                flow_bwd = padder.unpad(flow_pr[1]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
            else:
                flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_flow_bwd.png')
            save_vis_flow_tofile(flow_bwd, output_file)

        # Forward-backward consistency check
        if fwd_bwd_consistency_check:
            if inference_size is None:
                fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
                bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
            else:
                fwd_flow = flow_pr[0].unsqueeze(0)
                bwd_flow = flow_pr[1].unsqueeze(0)

            fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)  # [1, H, W] float

            fwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ.png')
            bwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_occ_bwd.png')

            Image.fromarray((fwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(fwd_occ_file)
            Image.fromarray((bwd_occ[0].cpu().numpy() * 255.).astype(np.uint8)).save(bwd_occ_file)

        if save_flo_flow:
            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[:-4] + '_pred.flo')
            frame_utils.writeFlow(output_file, flow)
