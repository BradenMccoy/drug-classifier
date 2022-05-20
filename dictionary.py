import sys
import getopt
import pandas as pd

def main(argv):
    inputfile = ''
    outputfile = ''

    try:
        opts = getopt.getopt(argv, "hi:o:", ["ifile=","ofile="])
    except getopt.GetoptError:
        print('macros.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    #interpret args
    for opt, arg in opts:
        if opt == '-h':
            print('macros.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        else:
            print('macros.py -i <inputfile> -o <outputfile>')
            sys.exit(1)
            
    #pandas dataframe
    data1 = pd.read_csv(inputfile, header=None)
    data1.columns = ['Id', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    data1.drop('Id', axis=1, inplace=True)
    #data.drop('Nscore', axis=1, inplace=True)
    #data.drop('Escore', axis=1, inplace=True)
    #data.drop('Oscore', axis=1, inplace=True)
    #data.drop('Ascore', axis=1, inplace=True)
    #data.drop('Cscore', axis=1, inplace=True)
    #data.drop('Impulsive', axis=1, inplace=True)
    #data.drop('SS', axis=1, inplace=True)
    
    data2 = data1
    
    #numbers matched with attribute values
    numbers = [-0.95197,-0.07854,0.49788,1.09449,1.82213,2.59171,0.48246,-0.48246,-2.43591,-1.73790,-1.43719,-1.22751,-0.61113,-0.05921,0.45468,1.16365,1.98437,-0.09765,0.24923,-0.46841,-0.28519,0.21128,0.96082,-0.57009,-0.50212,-1.10702,1.90725,0.12600,-0.22166,0.11440,-0.31685,-0.99804,0.99812,-0.99761,0.99729,-0.99676,0.99570,-0.99768,0.99720,-0.99791,0.99713,-0.94725,0.96167,-0.96699,0.96061]
    attributes = ["18-24","25-34","35-44","45-54","55-64","65+","Female","Male","Left school before 16 years","Left school at 16 years","Left school at 17 years","Left school at 18 years","Some college or university (no certificate or degree)","Professional certificate/diploma","University degree","Masters degree","Doctorate degree","Australia","Canada","New Zealand","Other","Republic of Ireland","UK","USA","Asian","Black","Mixed-Black/Asian","Mixed-White/Asian","Mixed-White/Black","Other","White","Not Neurotic","Neurotic","Not Extraverted","Extraverted","Not open to experience", "Open to experience","Not Agreeable", "Agreeable", "Not Conscientious", "Conscientious","Not Impulsive","Impulsive","Not Sensation Seeing","Sensation Seeing"]
    
    #go through the data and replace data in columns with matching stuff
    for (key, values) in data1.iteritems():
        
        #print("Key:", key)
        #print("Value:", value)
        #print()
        
        for i in range(len(numbers)):
            for j in range(len(values)):
                if (numbers[i] == values[j]):
                    data2[key, values[j]] = attributes[i]
                    break
                
    data2.to_csv(outputfile, index=False)
        

if __name__ == '__main__':
    main(sys.argv[1:])
