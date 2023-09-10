# Emotion_classifier
M.Sc. in Language Technology - M906 Programming for Language Technology II

The purpose of this project is to create and evaluate a system that recognizes real-time emotion from speech. 
1) Choose an emotion categorization problem (e.g., "calm"-"angry")
2) Select appropriate audio features (among MFCCs, spectral centroid & bandwidth)
3) Select and train a machine learning model (however simple), presenting results of its evaluation, not in real time (i.e., by the split training-test data process)
4)Save this model for loading and use in the real-time program you will build later.
5) Evaluate the model you saved above in real-time mode, but with the audio coming from audio file data (for testing in "lab conditions"). It is your choice how many files to evaluate it on and what metric to use. We chose to evaluate on a windows size of 2048
6) Run the model with your voice and print the results as you wish (e.g. on a graph or even as a printout on the console). Save the result in whatever format you consider sufficiently descriptive (e.g. screenshot or copy-paste from the console)

## User Instructions
Analytical instructions about the order to run the .py scripts are given in the **Realtime_SER_Report.pdf**

## Contributor Expectations
Extend the project for a  real-time multi-class classification system, by icluding more than 2 emotion classes in the project.

## References
"The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0

https://gitlab.com/language-technology-msc/programming-for-language-technology-ii2022-2023/programminglangtechii_c03/-/blob/main/4_run_ML_tests.py

https://gitlab.com/language-technology-msc/programming-for-language-technology-ii2022-2023/programminglangtechii_c05/-/blob/main/1a_file_playback.py

https://gitlab.com/language-technology-msc/programming-for-language-technology-ii2022-2023/programminglangtechii_c05/-/blob/main/5_mic_spetrogram.py

https://github.com/vibhash11/Emotion-Recognition-Through-Speech
