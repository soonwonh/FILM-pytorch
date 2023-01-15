from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import torch

from utils import to_gpu, save_checkpoint, load_checkpoint, metrics, log_image

def train(args, model, summary, optimizer, create_losses_fn, dataloader, eval_loop_fn=None, eval_datasets=None, resume=None):

    loss_functions = create_losses_fn(['l1', 'perceptual'])
    #metrics = create_metrics_fn()
    PSNR = PeakSignalNoiseRatio().cuda()
    SSIM = StructuralSimilarityIndexMeasure().cuda()


    model.train()
    global_step = 0
    for epoch in range(args.epoch):

        save_checkpoint(args, model, optimizer, epoch)
        metric_psnr, metric_ssim = 0, 0

        for i, batch in enumerate(tqdm(dataloader)):
            batch = to_gpu(batch)
            model.zero_grad()
            predictions = model(to_gpu(batch))
            optimizer.zero_grad()
            losses = []
            for (loss_function) in loss_functions:
                loss = loss_functions[loss_function](batch, predictions)
                losses.append(loss)
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            global_step+=1

            summary.add_scalar('train/loss', float(loss), global_step=global_step)

            if i % 100==0:
                log_image(batch, predictions, args, summary, epoch, i, global_step)
                with torch.no_grad():
                    psnr, ssim = metrics(predictions, batch, summary, PSNR, SSIM, global_step)
                    metric_psnr += psnr
                    metric_ssim += ssim

        summary.add_scalar('train/psnr_epoch', float(metric_psnr/(len(dataloader))*100), global_step=epoch)
        summary.add_scalar('train/ssim_epoch', float(metric_ssim/(len(dataloader))*100), global_step=epoch)

       
