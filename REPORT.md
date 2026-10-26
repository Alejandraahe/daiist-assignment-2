# Assignment 2 Report

- **Name**: Alejandra Hernández Espinosa
- **Student ID**: 20122
- **Email**: ahernandeze.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Business problem, and the cost of being wrong
This project looks at the cash centre of a bank in the Philippines. Counting machines put aside notes that look suspicious, and today a teller looks at each of these notes and decides by eye if it is fake. My model takes a photo of each flagged note and gives it a counterfeit score from 0 to 1. If the score is below a low threshold, the note passes and is accepted as genuine. If it is between the two thresholds, a teller checks it with a UV lamp, the watermark and the security thread. If it is at or above the high threshold, the note fails, the bank keeps it and sends it to the central bank. The positive class is counterfeit, so a "positive" means the model says the note is fake.

Each outcome has a cost. I used an exchange rate of about 65 pesos per euro. If a fake note passes, the bank loses its full face value, because it cannot pass the note on or get the money back. This goes from €0.31 for a 20-peso note to €0.77 for 50, €1.54 for 100, €3.08 for 200, €7.69 for 500 and €15.38 for 1000. If a genuine note is wrongly kept, I estimate a cost of €3. The customer complains, a supervisor checks the note again and returns it, and the bank loses some goodwill. That is about 15 to 20 minutes of staff time plus the damage to the relationship. When a fake is correctly kept, there is still a cost of €0.50 for the paperwork to hand it to the central bank. A genuine note that passes costs nothing. Sending a note to a person costs about €0.10, which is around 90 seconds of a teller's time at about €4 per hour. I assume the teller always gets it right.

Which mistake is worse depends on the note. For a 1000-peso note, missing a fake costs €15.38, about five times more than a false alarm at €3. For a 20-peso note it is the opposite: a false alarm costs €3, about ten times more than the €0.31 lost by missing the fake. On average, over the fakes in the test set, the face value is 312 pesos, about €4.79, so a missed fake costs about 1.6 times a false alarm. This means the bank should be strictest with high-value notes.

I also made two assumptions. First, that 5% of the notes the machines flag are really fake, because most flagged notes are genuine but old, folded or dirty. Second, that 2,000 flagged notes reach the centre every day. These numbers are my estimates, not real bank data.

## Dataset
I used the CCM 2024 dataset from Roboflow Universe, in the workspace thesis2024, project ccm-2024, version 2 (https://universe.roboflow.com/thesis2024/ccm-2024/dataset/2). It is an object detection dataset under the CC BY 4.0 licence, and I used it as it is, without forking it. Each image is a photo of one side of one peso note, and the note fills most of the picture.

There are twelve classes. Six are counterfeit notes (20, 50, 100, 200, 500 and 1000 pesos), five are genuine notes (20, 50, 200, 500 and 1000 pesos), and there is one extra class simply called "counterfeit".

This dataset fits the problem because it has both fake and genuine notes, labelled by value. The model can learn the difference between them, and I can link each missed fake to its face value, which my cost model needs.

Before choosing it, I rejected three other datasets. A Pakistani rupee dataset only had two classes for classification, and the brief needs at least three. A US dollar dataset had no versions, so it could not be downloaded without forking it, which is not allowed. Other peso datasets only labelled values and security features with no counterfeit class, or had just 73 images.

## What the dataset's composition implies
The dataset has 2,427 images: 1,767 for training, 440 for validation and 220 for testing. Each of the eleven main classes has 140 notes in training, 40 in validation and 20 in test. The extra "counterfeit" class has 217 boxes, all in training, and ten training images have no label. About 60% of training images and 55% of validation and test images show a fake. The notes are large: the typical box covers 58% of the image in training and 63% in validation and test.

This has several consequences. The classes are perfectly balanced, which helps training, but it is very different from real life, where I expect only 5% of flagged notes to be fake instead of 55%. For this reason I measure the model on fakes and on genuine notes separately, and then combine the results using the 5% rate.

There are only 20 test notes per class, so one single mistake would change that class's recall by 5 points. The results per class are therefore rough. Because the notes are so large, finding the note in the picture is easy, and the real task is deciding if it is fake or genuine. There are no genuine 100-peso notes at all, so the model has only seen fake 100s and would probably call a real one fake. The test set cannot show this. There are also small labelling problems: the extra "counterfeit" class only in training and the ten images with no label.

The most important finding is that the images are frames taken from 22 short videos. The file names show this. For example, "coun1000b-f14" is frame 14 of the video of a counterfeit 1000-peso note seen from the back. There are 22 videos, one for each mix of fake or genuine, value and side, with around 100 frames each, and every video appears in training, validation and test. So a test image is almost the same as the training images just before and after it. It also seems that only one physical note was filmed for each class.

Because of this, I cannot conclude that the model has learned what makes a note fake. It may only recognise those specific notes. The test result shows how well the model remembers, not how well it would work on new notes, in different light, with worn or folded notes, or with another camera.

## Finetuning and baselines
I used object detection because the dataset is labelled that way, and because a box shows the teller where the model looked. I chose YOLO11n, the smallest YOLO11 model, starting from the version trained on the COCO dataset. It is fast enough to run on a normal computer at a teller's desk, and the task does not need a bigger model.

I trained it on Google Colab with a free T4 GPU. I allowed up to 50 epochs, with an image size of 640, a batch size of 16 and seed 42. Training stops early if nothing improves for 15 epochs, and it stopped after 22 epochs, in about 12 minutes. The model already reached an mAP50 of 0.967 after the first epoch and 0.995 from the third.

I compared three models on the same 220 test images. My finetuned model reached an mAP50 of 0.995 and an mAP50-95 of 0.995, with a precision of 0.998 and a recall of 1.000. It caught all 120 fakes and rejected none of the 100 genuine notes. The original COCO model, used without any training on notes, scored 0, because COCO has no banknote class. Instead it found things like "person" (60 times), "clock" (23), "tie" (22) and "dog" (20) in 124 of the images. The simplest baseline, a model that predicts nothing, also scored 0, which means every note passes and every fake is accepted.

Beating the baselines was certain, since neither can output my classes. What surprised me was the size of the win and how fast it came, by the third epoch. I do not think this shows an excellent model. It is explained by the problem with the videos described above.

## Per-class evaluation and failure modes
Every class scored about the same. The six counterfeit classes had a precision between 0.996 and 1.000, a recall of 1.000 and an AP50 of 0.995. The five genuine classes had a precision between 0.995 and 1.000, a recall of 1.000 and an AP50 of 0.995. Each class had 20 test notes.

The confusion matrix, using boxes with a confidence of 0.25 or more, is a perfect diagonal. All 20 notes in each class were correct, no class was confused with another and no note was missed. Looking at each note, all fakes scored between 0.934 and 0.99, and all genuine notes scored 0.001 or less, with nothing in between.

No class carries the result and none pulls it down, since they are all equal. This in itself is suspicious, and again points to the near-identical frames.

Since the test set shows no mistakes, I can only describe the failures I expect but cannot measure. Genuine 100-peso notes were never in the data, so the model would probably call them fake. Each class seems to come from a single note, so a different real note could look unfamiliar to the model. In a real cash centre the light, the camera and the condition of the notes would be different, and the newer plastic 1000-peso notes look different from the paper ones. Finally, small notes are a problem for costs: wrongly rejecting a genuine 20-peso note costs €3, ten times the value of the note itself.

## Cost matrix and operating point
The costs for each type of error and where they come from are explained in the first section. I assumed a base rate of 5% fake notes.

To work out the cost per 1,000 notes, I first gave each note a counterfeit score, which is the highest confidence of any counterfeit box, or 0 if there is none. For each threshold, each note is passed, reviewed or failed and gets its cost. I then take the average cost of the fakes and the average cost of the genuine notes separately, and combine them as 5% of the fake cost plus 95% of the genuine cost, multiplied by 1,000.

I tested every threshold from 0.00 to 1.00 in steps of 0.01 on the validation set, looked for the lowest cost, and then reported the result on the test set. Every threshold from 0.01 to 0.93 gave the same lowest cost, because of the large gap between the scores of fakes and genuine notes. Instead of taking 0.01, which is very close to the genuine notes, I took the middle of this range, 0.47. This leaves the most room for error on both sides, which matters because real notes will not score as cleanly.

On the test set, with a 5% base rate, doing nothing and accepting every note costs €239.74 per 1,000 notes and stops no fakes. Having a person check every note costs €125 and stops all fakes. The model on its own at a threshold of 0.47 costs €25, and the model with a review band from 0.01 to 0.93 also costs €25. Both stop all fakes, keep no genuine notes and send no notes to a person. The cost is exactly €25 because the model makes no mistakes on the test set, so the only cost left is the €0.50 paperwork for each fake caught: 1,000 notes times 5% times €0.50.

I also checked how the results change with the base rate. Doing nothing costs about 4,794 times the base rate per 1,000 notes, while checking everything costs 100 plus 500 times the base rate. The two are equal at about 2.3%. Below that, accepting everything is cheaper than checking everything by hand. The model stays the cheapest option at every base rate I tested, from 0.5% to 50%, as long as it stays this accurate.

The assumption that would most change my answer is that real notes look like the test notes. They do not, because of the 22 videos. If the model starts missing fakes, which can cost up to €15.38 each, or rejecting genuine notes at €3 each, its cost will rise quickly.

## What a person still has to check
My design uses two thresholds. Notes that score below 0.01 look like the genuine notes and pass. Notes that score 0.93 or more look like the fakes and fail. Notes in between go to a person for review. I chose the widest band among the options that cost the same, so that anything the model is unsure about is checked by a teller.

On the test set, no notes fell inside the band, so there would be no reviews and no tellers needed. This is not realistic. It happens because the test images are almost copies of the training images, and real notes will sometimes get a score in the middle.

For planning, I used 2,000 notes a day, 90 seconds per check and 6.5 working hours per teller. If 5% of notes go to review, that is 100 checks and 2.5 hours a day, or about 0.4 of a teller. At 10% it is 200 checks, 5 hours and about 0.8 of a teller. At 20% it is 400 checks, 10 hours and about 1.5 tellers. Checking every note would mean 2,000 checks, 50 hours and about 7.7 tellers a day.

Compared with the alternatives, doing nothing costs €240 per 1,000 notes and lets every fake through. Having a person check everything costs €125 per 1,000 notes and needs 7.7 tellers a day. The model alone costs €25 on the test set, but it has no safety net if it is wrong in real use. My design, the model with a review band, also costs €25 on the test set, and it sends uncertain notes to a person, which protects the bank from the gap between the test set and reality. If 10% of notes go to review, the extra cost is only about €10 per 1,000 notes, for a total of about €35.

I recommend starting with one teller on review and running the model in "shadow mode" for a few weeks, where the teller still checks every note. This would show the real share of notes sent to review and the real error rates, and the band could then be adjusted.

## Limitations & next steps
The biggest limitation is that the images are frames from 22 videos, split at random, so the test images are almost copies of the training images. With more time, I would split the data by video, for example training on the front side of each note and testing on the back, or I would film new notes separately for testing.

It also seems that only one physical note was filmed for each class. I would collect many different real and fake notes for each value so the model learns what makes a note fake, not what one note looks like.

There are no genuine 100-peso notes in the data. I would add them, and until then I would always send 100-peso notes to a person.

The labels have small problems: an extra "counterfeit" class that only appears in training, and ten images with no label. I would clean the labels and train the model again.

The base rate of 5% and the costs are my own estimates. I would replace them with the bank's real data. The sliders in the dashboard already show how the results change when these numbers move.

I use one threshold for all notes, even though missing a fake costs much more on high-value notes. I would use stricter thresholds for 500 and 1000-peso notes.

Finally, the photos were taken in controlled conditions, not at a real teller's desk with different light, cameras and worn notes. I would run a shadow-mode pilot before letting the model decide on its own.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment. Claude wrote the code
(banknotes.py, train.py, app.py and the Colab notebook), helped me search
Roboflow Universe for a suitable dataset, and explained the results,
including the finding that the dataset is built from frames of 22 videos.
The choice of problem, the cost and base-rate assumptions, and the
thresholds were discussed with Claude, and I am responsible for them. I ran
the training on Google Colab myself, checked the dashboard, and wrote this
report from Claude's explanations of the concepts and results.
