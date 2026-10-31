# L0 OSNet Retrieval Benchmark Decision

## Recommendation: Market-1501, conditional on local research-only use

Use the official Market-1501 query and bounding_box_test gallery for a
cross-dataset image-retrieval sanity benchmark. OSNet x0.25 here was trained on
MSMT17; Market-1501 test identities are a distinct benchmark split. Do not
train or fine-tune on Market-1501.

The dataset readme describes 3,368 query images and 19,732 test gallery images
across six cameras, and states that it is for research only, must not be
distributed, and must not be used commercially. This is a purpose-limited use
statement, not an OSI-style open license. Proceed only for local, non-commercial
research and keep the dataset outside Git. The historical author download
location is not clearly maintained; before acquiring files, confirm the
specific source and terms are acceptable. No images were downloaded in this
task.

Sources: [Market-1501 paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)
and [a preserved copy of the dataset readme](https://github.com/niruhan/market1501/blob/master/Market-1501-v15.09.15/readme.txt).
The readme copy is not an official distribution source; it is cited here only
for the published usage notice and split inventory.

## Benchmark protocol

1. Use query/ as queries and bounding_box_test/ as gallery; do not use
   bounding_box_train/ to fit or tune the encoder.
2. Decode each image as RGB, resize to 256×128, convert to float [0,1], apply
   ImageNet normalization, and run the verified OSNet adapter. L2-normalize
   output embeddings.
3. Rank gallery images by cosine similarity. Exclude gallery images with the
   same identity and same camera as the query; ignore junk IDs (-1/background)
   and the dataset's junk matches according to the official protocol.
4. Report CMC Rank-1/5/10, mean average precision, and recall@1/5/10. Include
   valid-query count and identities retained after junk filtering.
5. Do not adjust thresholds or the model using these test results. Report
   descriptive results only; this fixed test split does not validate CrossCam
   tracklet aggregation or cross-camera association.

The model preprocessing is sourced from the [upstream Torchreid transforms](https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/data/transforms.html):
ImageNet mean (0.485, 0.456, 0.406) and standard deviation
(0.229, 0.224, 0.225). Reports must identify the checkpoint hash and adapter
source commit.

## Status and next gate

**Not evaluated:** no Market-1501 images or crops were present in the checkout,
and no dataset acquisition was authorized by the current plan. Separate
approval is required before downloading benchmark images. After authorization,
verify the archive source, terms/readme, file inventory, and hashes locally;
then run the protocol without uploading images, crops, or embeddings.

Alternatives considered: MSMT17 is unsuitable as an independent test because
it is the checkpoint training domain and its distribution terms are
restricted; DukeMTMC-reID is not recommended because it was withdrawn
following privacy concerns; PRW has research-only conditions and a person-search
task protocol; CUHK03 has restricted access/use conditions and split variants
that complicate a lightweight first retrieval gate. Market-1501 is the most
direct compact candidate, subject to confirming research-only use and an
acceptable acquisition source.
