
RCL-Unet predicts the reactive center loop for Serpins. 


Run RCLPrediction.py {input}  -m {model}   -outpus

input: fasta formate proteins sequence, with sequence ID immediatedly follow >

output:fasta formate file of RCL sequencces. on the > definition line, is Sequence ID, RCL location (e.g. rcl:350-370), a score; the sequence are the RCL sequence predicted.  
    If no RCL was identified, the sequence space will have the line "not identified"