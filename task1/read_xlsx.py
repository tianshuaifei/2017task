import xlrd
import codecs
import task1.helper as voc
def read(data_path):
    da=voc.Vocab()
    da.load_vocab_from_file('data/vocab.txt')
    data = xlrd.open_workbook(data_path)
    file = codecs.open("data/train.txt", "w", "utf-8")
    tables = data.sheets()
    for table in tables:
        nrows = table.nrows
        i=0
        while i < nrows:
            rows = table.row_values(i)
            da.add_word(rows[0])
            da.add_word(rows[1])
            file.write(table.name+"\t"+rows[0]+"\t"+rows[1]+"\n")
            i=i+1
    da.save_vocab("data/vocab.txt")
    file.close()

if __name__ == '__main__':
    data_path='data/tasksampledata01.xlsx'
    read(data_path)