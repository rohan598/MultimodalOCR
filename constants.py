RANKINGS_PROMPT = """You are a helpful following assistant whose goal is to select the preferred (least wrong) output for a given instruction.
Answer the question by printing only a single choice from ["Response (a)", "Response (b)", "Response (c)"] (without quotes) corresponding to the correct answer with no other text.

## Annotation Guideline
In this task, we will ask you to select the preferred output AI model's responses to instructions.

You will read a examples, which are composed of the following:

1. an Instruction we give to the AI system
2. OCR Text of the image
3. a Ground Truth Response for the Instruction
4. The output from the first AI system 
5. The output from the second AI system

You have to select from one of the option
1. Response (a), the output from the First AI system
2. Response (b), the output from the Second AI system
3. Response (c), the output from both AI systems

Your task is to decide which response is better for each example. 

Accuracy: The output sentence should be factually consistent with the Ground Truth Response.

Coherence: The output sentence should be easy to understand and free of 
grammatical errors when read on its own.

Non-repetitive: DO NOT Prefer long output sentence if it is not factually consistent with the Ground Truth Response.The output sentence by AI system should not be preferred if it repeats the text in the instruction but does not answer the instruction with Accuracy.

In extractive instructions like who, when, count and so on, Please Focus on matching the value of the entity like person, time, number than the actual phrasing of the response.

In summative instructions like summarize, purpose, understand, Please Focus on matching the gist conveyed in Output from First AI system and Output from Second AI system to the Ground Truth Response. 

You do not provide Human Explaination of the answer. Human Explaination only provided in examples to help build your reasoning.

You should answer using only Response (a), Response (b) or Response (c)

## Annotation Example
To help you understand the annotation task, we provide some examples below.
I will give an explanation for the correct answer, but you should only answer with the preferred output.

### Example 1

#### Instruction 1:
What is the duration of the first track?

#### OCR Text of the image
SA07:48\n        Lil Uzi Vert\n                       The Perfect Luv Tape\n                       10 tracks-2044comments\n                       185847hits-8737likes\n                             Download all\n  01 Do WhatI Want Prod.By Maaly Raw &Don\n  Cannon]                                       2:59\n  Lil Uzi Vert-The Perfect Luv Tape\n  02 Of Course We Ghetto Flowers [Prod.By\n                                                Maaly Raw4:23\n  Lil Uzi Vert-The Perfect Luv Tape\n  03 OriginalUzi4 Of UsProd.By Maaly Raw\n                                                2:47\n  Lil Uzi Vert-The Perfect Luv Tape\n  04 Money Mitch [Prod.By Zaytoven\n                                                4:15\n  Lil Uzi Vert-The Perfect Luv Tape\n  05 SideLine Watching Hold Up)[Prod.By\n  Zaytoven]                                     3:12\n  Lil Uzi Vert-The Perfect Luv Tape\n  06Alfa Romeo AW30 Prod.By DP Beatz\n                                                2:34\n       FOORFEARLESS\n         SEETHEFEARLESS                    912831-10/102016

#### Ground Truth Response 1:
The duration of the first track is 2 minutes and 59 seconds.

#### Output of First AI system for Example 1:
2:59

#### Output of Second AI system for Example 1:
The duration of the first track is 3 minutes.

#### Answer for Example 1:
Response (a)

#### Human Explaination for Example 1: Indeed, Response (a) as the First AI system gets duration value correct.

### Example 2

#### Instruction 2:
Count the number of vowels in the word.

#### OCR Text of the image
      ijecu

#### Ground Truth Response 2:
2

#### Output of First AI system for Example 2:
Two

#### Output of Second AI system for Example 2:
2

#### Answer for Example 2:
Response (c)

#### Human Explaination for Example 2: Indeed, Response (c) as the both AI system get the right answer

### Example 3

#### Instruction 3:
What is the date and time when the article was last updated?

#### OCR Text of the image
SUBSCRIBE((o) RADIOO LIVEO LIGHT\n Global>                                                               WorldCanadaLocalPoliticsMoneyHealthEntertainmentLifestyleWatchQV\n   NEWS\n Transit increases to take effect Jan. 1 in\nWinnipeg\n    By caswelllogan : Global News.\n    Posted December 31, 2017 1:41 pm : Updated January 2, 2018 1:55 pm\n                         WINNIPEG TRANSIT\n                        SR1724\n                                                                    D AdChoices

#### Ground Truth Response 3:
January 2, 2018 1:55 pm

#### Output of First AI system for Example 3:
The article was last updated on January 1, 2023 at 12:00 AM.

#### Output of Second AI system for Example 3:
The article was last updated on January 1, 2018 at 12:00 PM.

#### Answer for Example 3:
Response (b)

#### Human Explaination for Example 3: Indeed, Response (b) the Second AI system is closer to the right response

## Annotation starts below
Now is your turn. I will give you an example.
You should read the example and then select the preferred answer by saying only Response (a), Response (b), or Response (c) as formatted above without Human Explanation.

## Example 4

### Instruction 4:
{instruction}

#### Ground Truth Response 4:
{gt_answer}

### Output of First AI system for Example 4:
{output_1}

### Output of Second AI system for Example 4:
{output_2}

## Answer for example 4:
"""