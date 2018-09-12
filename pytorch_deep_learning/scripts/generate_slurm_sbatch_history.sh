
# To generate history of all previous batches in this month
sacct -S `date --date "last month" +%Y-%m-%d` -o "Submit,JobID,JobName,Partition,NCPUS,State,ExitCode,Elapsed,CPUTime,MaxRSS" 
