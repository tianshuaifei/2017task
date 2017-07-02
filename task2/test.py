import configparser
def main():
    cfg = configparser.ConfigParser()
    cfg['bitbucket.org'] = {}
    cfg['bitbucket.org']['User']="1"
    cfg['bitbucket.org']['User'] = 'hg'
if __name__ == '__main__':
    main()