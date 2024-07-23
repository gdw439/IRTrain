def load_tsv(file_path):
    ''' 
        given a *.tsv file, load each line in yield format
    '''
    import csv, sys
    csv.field_size_limit(sys.maxsize)
    with open(file_path, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)
        for [idx, text] in reader:
            yield text