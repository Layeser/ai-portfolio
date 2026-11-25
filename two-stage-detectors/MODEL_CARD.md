# R-CNN, Fast R-CNN and Faster R-CNN — PASCAL VOC 2007 weights

Weights of the three detectors re-implemented in this project, all on the same ImageNet-pretrained ResNet-50 base,
trained on VOC 2007 trainval (minus a 300-image holdout used for every choice) and evaluated once on the 4,952 images
of VOC 2007 test (mAP@0.5, VOC 2007 11-point metric).

| File | Model | VOC 2007 test mAP | GPU time / image |
|---|---|---|---|
| `faster_rcnn_voc2007_fp16.pt` | Faster R-CNN (learned proposals, RPN) | **71.7 %** | 0.08 s |
| `fast_rcnn_voc2007_fp16.pt` | Fast R-CNN (Selective Search proposals) | 67.2 % | 0.29 s + Selective Search |
| `rcnn_voc2007_fp16.pt` | R-CNN (one CNN pass per region) | 62.5 % | 1.01 s + Selective Search |

Weights are stored in float16 to halve the download (model weights only, no optimiser state); they are loaded into
the float32 model as usual. The `eval_*_test.json` files are the full test results of the original float32
checkpoints.

Check on the first 300 test images (mAP@0.5, VOC 2007 metric), float32 → float16: Faster R-CNN 0.741 → 0.749,
Fast R-CNN 0.713 → 0.714, R-CNN 0.645 → 0.649. The differences come from rounding and stay within one point; the
full-test scores above are those of the float32 checkpoints.

**Use it** (from `two-stage-detectors/`):

```bash
make download-weights        # -> checkpoints/<model>/<model>_voc2007_fp16.pt, SHA-256 verified
python -m twostage.evaluate --model faster_rcnn --weights checkpoints/faster_rcnn/faster_rcnn_voc2007_fp16.pt --max-images 300
```

Fast R-CNN and R-CNN need the Selective Search proposals (`make proposals`); Faster R-CNN does not. Code, protocol
and departures from the papers: [two-stage-detectors/README.md](https://github.com/Layeser/ai-portfolio/tree/main/two-stage-detectors).
Trained on PASCAL VOC 2007 (images under their Flickr terms of use).
