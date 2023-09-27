# severityPrediction

## TODO
- [ ] Preprocess the full data
	- [ ] Efficient implementation
- [ ] Analyse samples of the results
	- [ ] Saw duplicated texts
	- [ ] What is the size of the vocabulary
- [x] Use as text-generation or text-classification
- [x] Remove the explanation
- [x] Llama2 incoherent answers

## Pipeline

```tikz
\begin{document}
\tikzstyle{abstract}=[draw=black, rectangle, text centered, align=center]

\def\distShift{0.2cm}
\def\fitdistShift{0.4cm}
\begin{tikzpicture}
    \usetikzlibrary{positioning,fit,calc,matrix}
    
    \node [abstract, draw=white] (input)  {Input\\(sometimes with code)};
    \node [abstract, below=\distShift of input] (summary)  {Summary};
    \node [abstract, draw=white, text width=, below=\distShift of summary] (or)  {OR};
    \node [abstract, below=\distShift of or] (description)  {Description};
    \node [draw=black, fit={(input) (summary) (or) (description)}] (fittedInput) {};

    
    % Model
    \node [abstract, draw=white, below=0.6cm of fittedInput] (model)  {Model};
    % Baseline
    \node [abstract, draw=white, below=\distShift of model] (baselines)  {Baselines};
    \node [abstract, below=\distShift of baselines] (knn)  {KNN};
    \node [abstract, below=\distShift of knn] (svm)  {SVM};
    \node [abstract, below=\distShift of svm] (bayes)  {Naïve Bayes};
    \node [draw=black, fit={(baselines) (knn) (svm) (bayes)}] (fittedBaselines) {};
    
    \node [abstract, draw=white, below=\fitdistShift of fittedBaselines] (modelAnd)  {AND};
    
    \node [abstract, draw=white, below=\fitdistShift of modelAnd] (deeplearning)  {Deep learning};
    \node [abstract, below=\distShift of deeplearning] (llama)  {LLAMA};
    \node [abstract, ultra thin, below=\distShift of llama] (lstm)  {LSTM};
    \node [draw=black, fit={(deeplearning) (llama) (lstm)}] (fittedDeeplearning) {};
    
    \node [draw=black, fit={(model) (fittedBaselines) (fittedDeeplearning)}] (fittedModel) {};
    
    
    \node [abstract, draw=white, below=0.8cm of fittedDeeplearning] (output)  {Output};
    \node [abstract, below=\distShift of output] (severity)  {Severity level\\major, critical, ...};
    \node [abstract, draw=white, below=\distShift of severity] (severityOr)  {OR};
    \node [abstract, below=\distShift of severityOr] (severityBin)  {Binary severity\\(severe VS not severe)};
    \node [draw=black, fit={(output) (severity) (severityOr) (severityBin)}] (fittedOutput) {};

    % arrows
    \draw[->] (fittedInput) -- (fittedModel);
    \draw[->] (fittedModel) -- (fittedOutput);
\end{tikzpicture}
\end{document}
```

```tikz
\begin{document}
\tikzstyle{abstract}=[draw=black, rectangle, text centered, align=center]

\def\distShift{0.2cm}
\def\fitdistShift{0.4cm}
\begin{tikzpicture}
    \usetikzlibrary{positioning,fit,calc,matrix}
    
    \node [abstract, draw] (csv)  {CSV file};
    
    \node [abstract, draw=white, below=\fitdistShift of csv] (input)  {Data\\(sometimes with code)};
    \node [abstract, below=\distShift of input] (summary)  {Summary};
    \node [abstract, draw=white, text width=, below=\distShift of summary] (or)  {OR};
    \node [abstract, below=\distShift of or] (description)  {Description};
    \node [draw=black, fit={(input) (summary) (or) (description)}] (fittedInput) {};
    
    \node [abstract, draw=white, below=0.6cm of description] (huggingfacePip)  {Custom Huggingface pipeline};
    \node [abstract, draw=black, below=0.2cm of huggingfacePip] (template)  {Template \\Insert data \& instruction};
    \node [abstract, draw=black, right=3.5cm of template, text width=5cm, align=left] (templateDetail)  {\emph{
    Below is an instruction that describes a task. Write a response that appropriately completes the request.\\
\#\#\# Instruction:\\
Categorize the bug report into one of 2 categories:\\
0 = NOT SEVERE\\
1 = SEVERE\\
\#\#\# Input:\\
\{data\}\\
\#\#\# Response:
    }};
    \node [abstract, draw=black, text width=5cm, above=0.2cm of templateDetail] (preprompt)  {Optional preprompt\\\emph{Always anwer with one token. Do not give any explanation. Use only 0 or 1 and one token. Skip any politeness answer. You have only one word available}};
    
    \node [abstract, draw=white, below=\distShift of template] (or)  {OR};
    \node [abstract, draw, left=\distShift of or] (pipClas)  {TextClassificationPipeline};
    \node [abstract, draw, right=\distShift of or] (pipGen)  {TextGenerationPipeline};
    \node [abstract, draw, below=\distShift of pipClas] (score)  {$0 \hookrightarrow 1$ score};
    \node [abstract, draw, below=\distShift of pipGen] (textGen)  {Score categorization text};
    \node [abstract, draw, below=\distShift of textGen] (textGenInterp)  {Extract \\SEVERE\\ or\\ NOT SEVERE};
    \node [draw=black, fit={(huggingfacePip) (template) (or) (pipClas) (pipGen) (score) (textGen) (textGenInterp)}] (fittedHuggingfacePip) {};
    
    
    \draw[->] (csv) -- (fittedInput);
    \draw[->] (fittedInput) -- (fittedHuggingfacePip);
    \draw[->] (template) -- (templateDetail);
    \draw[->] (pipClas) -- (score);
    \draw[->] (pipGen) -- (textGen);
    \draw[->] (textGen) -- (textGenInterp);
\end{tikzpicture}
\end{document}
```

