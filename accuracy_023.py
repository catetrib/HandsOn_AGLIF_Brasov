from copy import copy
import math
import multiprocessing
'''
017
 - check input parameters
 - STSimM function: exposes mean weights to the user
018
- parallelized
- check for zero amplitude silent interval
- skip subinteval calculations if paramC = 1
019
- lambda default as kreutz 2017
- un solo weight
020
- sistemati i controlli su paramC
- se weigth = 1 fa solo un calcolo
022
- introdotto il calcolo del caso particolare a 0 spike del treno 1 (calcolaTNFP e calcolaIndiciDueSeriePlotSilenti)
- introdotto il valore Indeterminato per gli indici
- introdotta la stringa di output con il valore di Lambda
'''
def calcolaTPFPFN(tempiNumerici,ti,tiOld,tiNew,alpha,lamb):
    #calculates number of TruePositive, FalsePositive False Negative comparing values of serie2 with respect to a single value of serie1 and its neighbors. ti, tiOld and tiNew refer to serie1 times. 
    #Returns TP, FP,FN and the interval used 

    truePositiveTmp=0
    falsePositiveTmp = 0
    falseNegativeTmp=0

    #consideriamo ciascun semi intervallo al max di ampiezza lambda
    intervallo = [0,0]
    if alpha*(ti-tiOld)<lamb:
        intervallo[0]=ti-alpha*(ti-tiOld)
    else:
        intervallo[0]=ti-lamb
    if alpha*(tiNew-ti)<lamb:
        intervallo[1]=ti+alpha*(tiNew -ti)
    else:
        intervallo[1]=ti+lamb

    #conta il numero di spike numerici nell'intervallo
    result=len([i for i in tempiNumerici if (i>=intervallo[0] and i<=intervallo[1])])
    if result>=1: #true positive
        truePositiveTmp=truePositiveTmp+1
        if result>1: #se ce ne sono più di uno, dal secondo in poi sono false positive
            falsePositiveTmp=falsePositiveTmp+result-1
    else: #false negative: non ci sono spike numerici
        falseNegativeTmp=falseNegativeTmp+1;

    #print(intervallo)        
    return truePositiveTmp,falsePositiveTmp,falseNegativeTmp,intervallo

def calcolaTNFP(tempiNumerici,intervalloStart,intervalloEnd,alpha,lamb):
    #calculates number of TrueNegative and FalsePositive  in a silent interval [intervallStart,intervalloEnd]
    #Returns TN, FP and the interval used 
    trueNegativeTmp=0
    falsePositiveTmp = 0

    #calcolo TrueNegative --------------------------------------------
    #consideriamo l'intervallo tra il tempo numerico i ed il tempo i+1 calcolato dalla funzione calcolaTPFPFN
    intervalloSilente=[intervalloStart,intervalloEnd]
    ampiezza=intervalloSilente[1]-intervalloSilente[0]
    
    coeffC = alpha
    subintervalToReturn=[]
    
    # if paramC = 1: use just 1 interval, no subintervals needed
    if coeffC == 1:
        numeroSubint=1;
        #conta il numero di spike numerici nel sotto intervallo
        subinterval=intervalloSilente;
        res=len([i for i in tempiNumerici if (i>=subinterval[0] and i<=subinterval[1])])
        if res==0:
            trueNegativeTmp=trueNegativeTmp+1
        else:
            falsePositiveTmp=falsePositiveTmp+res
        subintervalToReturnP=copy(subintervalToReturn)
        subintervalToReturn.append(intervalloSilente)        
    
    else: #calculates amplitude
        soglia=coeffC*2*lamb
        if ampiezza<=2*lamb:
            numeroSubint=1
            lambdaRicalcolato=ampiezza
        else:
            if 2*lamb<ampiezza<=soglia:
                numeroSubint=math.floor((ampiezza/(2*lamb)))
                lambdaRicalcolato=2*lamb
            else:
                numeroSubint=math.floor((soglia/(2*lamb)))
                lambdaRicalcolato=ampiezza/math.floor((soglia/(2*lamb)))
    
        
        if ampiezza<lamb:
    	    #lambdaRicalcolato smaller than lambda => 1 interval
            numeroSubint=1;
            #conta il numero di spike numerici nel sotto intervallo
            subinterval=intervalloSilente;
            res=len([i for i in tempiNumerici if (i>=subinterval[0] and i<=subinterval[1])])
            if res==0:
                trueNegativeTmp=trueNegativeTmp+1
            else:
                falsePositiveTmp=falsePositiveTmp+res
            subintervalToReturnP=copy(subintervalToReturn)
            subintervalToReturn.append(intervalloSilente)
        #ELSE: lambdaRicalcolato greater than lambda => procedure for TNFP
        else:
            if numeroSubint>1:
                for j in range(numeroSubint):
                    subinterval=[intervalloSilente[0]+(j)*lambdaRicalcolato,intervalloSilente[0]+(j+1)*lambdaRicalcolato]
                    subintervalToReturn=copy(subintervalToReturn)
                    subintervalToReturn.append(subinterval)
                    #conta il numero di spike numerici nel sotto intervallo
                    res=len([i for i in tempiNumerici if (i>=subinterval[0] and i<=subinterval[1])])
                    if res==0:
                        trueNegativeTmp=trueNegativeTmp+1
                    else:
                        falsePositiveTmp=falsePositiveTmp+res
                #conteggio dell'ultimo intervallo
                if (intervalloSilente[0]+numeroSubint*lambdaRicalcolato)<intervalloSilente[1]:
                    subinterval=[(intervalloSilente[0]+numeroSubint*lambdaRicalcolato),intervalloSilente[1]]
                    #print(subinterval)
                    subintervalToReturn=subintervalToReturn.append(subinterval)
                    res=len([i for i in tempiNumerici if (i>=subinterval[0] and i<=subinterval[1])])
                    if res==0:
                        trueNegativeTmp=trueNegativeTmp+1
                    else:
                        falsePositiveTmp=falsePositiveTmp+res
            else:
                # just 1 subinterval
                subinterval=intervalloSilente;
                res=len([i for i in tempiNumerici if (i>=subinterval[0] and i<=subinterval[1])])
                if res==0:
                    trueNegativeTmp=trueNegativeTmp+1
                else:
                    falsePositiveTmp=falsePositiveTmp+res
                subintervalToReturnP=copy(subintervalToReturn)
                subintervalToReturn.append(intervalloSilente)                        
                        

    return trueNegativeTmp,falsePositiveTmp,subintervalToReturn

#♣def calcolaIndiciDueSeriePlotSilenti(maxTime,tempiSperimentaliOrig,tempiNumerici,paramC,lamb,omega,result_queue):
def calcolaIndiciDueSeriePlotSilenti(maxTime,tempiSperimentaliOrig,tempiNumerici,paramC,lamb,omega):
    #Calculates Accuracy, Precision, F1 score and Recall given two lists of ordered spike times. Assumes that the signal starts at 0 ms.

    # parameters check
    paramError=False
    if omega>0.5:
        #print("omega max admissible value is 0.5; try again")
        paramError=True
    if isinstance(paramC, int)==False:
        #print("paramC must be integer; try again")
        paramError=True
    if paramC<=0:
        #print("paramC must be >0; try again")
        paramError=True
        
    if paramError==False:
        #NB: ai tempi sperimentali vengono aggiunti due limiti [0,maxTime] prima di inserirli nella funzione
        startTime=0 #tempiSperimentaliOrig[0]-10
        endTime=maxTime #tempiSperimentaliOrig[[-1]]+10*
        tempiSperimentaliTmp=copy(tempiSperimentaliOrig)
        tempiSperimentaliTmp.insert(0,startTime)
        tempiSperimentali=copy(tempiSperimentaliTmp)
        tempiSperimentali.insert(len(tempiSperimentaliTmp),endTime)
        
        # print(tempiSperimentaliOrig)
        # print(tempiSperimentaliTmp)
        # print(tempiSperimentali)
        
        trueNegative = 0;
        truePositive=0;
        falsePositive = 0;
        falseNegative=0;
        intervalliTempiSperimentali={}
        intervalliTempiSperimentali[0]=[tempiSperimentali[0],tempiSperimentali[0]] #ampiezza = 0, non lo usiamo
        intervalliTempiSperimentali[len(tempiSperimentali)-1]=[tempiSperimentali[len(tempiSperimentali)-1],tempiSperimentali[len(tempiSperimentali)-1]] #ampiezza = 0
    	
        #Tutti i valori da 1 a -1
        if len(tempiSperimentali)>2:
    	    #* 1) calcolo TPFPFN
            for i in range(1,len(tempiSperimentali)-1):
                ti=tempiSperimentali[i]            
                tiOld=tempiSperimentali[i-1]
                tiNew=tempiSperimentali[i+1]
                truePositiveTmp,falsePositiveTmp,falseNegativeTmp,intervallo=calcolaTPFPFN(tempiNumerici,ti,tiOld,tiNew,omega,lamb)
                truePositive=truePositive+truePositiveTmp
                falsePositive=falsePositive+falsePositiveTmp
                falseNegative=falseNegative+falseNegativeTmp
                intervalliTempiSperimentali[i]=intervallo
    
            # 2) calcolo TNFP *
            intervalliSilenti=[]
            
            # primo tempo silente
            intervalloStart=intervalliTempiSperimentali[0][1]
            intervalloEnd=intervalliTempiSperimentali[1][0]
            if intervalloStart -intervalloEnd != 0:
                trueNegativeTmp,falsePositiveTmp,intervalliSilentiSingleStep=calcolaTNFP(tempiNumerici,intervalloStart,intervalloEnd,paramC,lamb)
                intervalliSilenti.append(intervalliSilentiSingleStep)
                trueNegative=trueNegative+trueNegativeTmp
                falsePositive=falsePositive+falsePositiveTmp
            # tempi silenti successivi
            for i in range(1,len(tempiSperimentali)-1):
                intervalloStart=intervalliTempiSperimentali[i][1]
                intervalloEnd=intervalliTempiSperimentali[i+1][0]
                if intervalloStart -intervalloEnd != 0:
                    trueNegativeTmp,falsePositiveTmp,intervalliSilentiSingleStep=calcolaTNFP(tempiNumerici,intervalloStart,intervalloEnd,paramC,lamb)
                    intervalliSilenti.append(intervalliSilentiSingleStep)
                    trueNegative=trueNegative+trueNegativeTmp
                    falsePositive=falsePositive+falsePositiveTmp
            
            #intervalliSilenti=Flatten[intervalliSilenti,1];
    
            myAccuracy=(truePositive+trueNegative)/(truePositive+falsePositive+falseNegative+trueNegative)*100
            if truePositive+falsePositive!=0:
                myPrecision=(truePositive)/(truePositive+falsePositive)*100
            else:
                myPrecision=1000 #ComplexInfinity
            myF1score=(2*truePositive)/(2*truePositive+falsePositive+falseNegative)*100
            if (truePositive+falseNegative)!=0:
                myRecall=(truePositive)/(truePositive+falseNegative)*100
            else:
                myRecall=1000
        # ELSE: se non ci sono spike sperimentali
        else:
            intervalloStart = 0
            intervalloEnd = maxTime
            trueNegative,falsePositive,intervalliSilentiSingleStep=calcolaTNFP(tempiNumerici,intervalloStart,intervalloEnd,paramC,lamb)
            
            myAccuracy=(truePositive+trueNegative)/(truePositive+falsePositive+falseNegative+trueNegative)*100
            
            if truePositive+falsePositive!=0:
                myPrecision=(truePositive)/(truePositive+falsePositive)*100
            else:
                myPrecision=None #Indeterminate
            
            if (falsePositive)!=0:
                myF1score=(2*truePositive)/(2*truePositive+falsePositive+falseNegative)*100
            else:
                myF1score=None
            
            myRecall=None
            
        print('myAccuracy','myPrecision','myRecall','myF1score')
        results=[myAccuracy,myPrecision,myRecall,myF1score]
        #result_queue.put(results)
        return(results)
    
def getLambda(treno1,treno2,endTime):
    # lambda is a function of trains' ISI
    isiTreno1 = []
    isiTreno2 = []
    startTime=0
    
    if len(treno1)==1:
        isiTreno1.append((min(treno1[0]-startTime,endTime-treno1[0])))
    else:
        for i in range(len(treno1)-1):
            isiTreno1.append(treno1[i+1]-treno1[i])
            
    if len(treno2)==1:
        isiTreno2.append((min(treno2[0]-startTime,endTime-treno2[0])))
    else:
        for i in range(len(treno2)-1):
            isiTreno2.append(treno2[i+1]-treno2[i])
    
    lamb = 1/4 * math.sqrt(math.fsum([x**2 for x in isiTreno1] + [x**2 for x in isiTreno2]) / (len(isiTreno1)+len(isiTreno2)))
    print('lambda auto: ',lamb)    
    return(lamb)
     

def STSimM(endTime,treno1,treno2,paramC,lamb,omega,weight):
    skipCalculation = False

    if paramC == 'auto':
        paramC = 1
    if omega == 'auto':
        omega = 0.5    
    
    weight2 = 1- weight        
        

    # empty trains
    if len(treno1)==0 and len(treno2)==0:
        # no need to calculate indexes as accuracy = 1 and Precision, Recall, Fscore = None
        [myAccuracy,myPrecision,myF1score,myRecall] = [100.0,None,None,None]
        skipCalculation = True
        
    if skipCalculation == False:    
        # PERFORMANCE
        if weight == 1:
        
            if lamb == 'auto':
                lamb = getLambda(treno1,[],endTime)
        
            myAccuracy,myPrecision,myF1score,myRecall = calcolaIndiciDueSeriePlotSilenti(endTime, treno1, treno2, paramC, lamb, omega)
            
            
        # SIMILARITY
        else:
                
            if lamb == 'auto':
                lamb = getLambda(treno1,treno2,endTime)

            myAccuracy1,myPrecision1,myF1score1,myRecall1 = calcolaIndiciDueSeriePlotSilenti(endTime, treno1, treno2, paramC, lamb, omega)
            myAccuracy2,myPrecision2,myF1score2,myRecall2 = calcolaIndiciDueSeriePlotSilenti(endTime, treno2, treno1, paramC, lamb, omega)
        
            myAccuracy=(weight*myAccuracy1+weight2*myAccuracy2)/(weight+weight2)
            myPrecision=(weight*myPrecision1+weight2*myPrecision2)/(weight+weight2)
            myF1score=(weight*myF1score1+weight2*myF1score2)/(weight+weight2)
            myRecall=(weight*myRecall1+weight2*myRecall2)/(weight+weight2)
    
    return myAccuracy,myPrecision,myF1score,myRecall



        # result_queue1 = multiprocessing.Queue()
        # result_queue2 = multiprocessing.Queue()
        # process1 = multiprocessing.Process(target=calcolaIndiciDueSeriePlotSilenti,args=(endTime, treno1, treno2, paramC, lamb, omega,result_queue1))
        # process1.start()
        # process2 = multiprocessing.Process(target=calcolaIndiciDueSeriePlotSilenti,args=(endTime, treno2, treno1, paramC, lamb, omega,result_queue2))
        # process2.start()
        # process1.join()
        # [myAccuracy1,myPrecision1,myF1score1,myRecall1] = result_queue1.get()
        # process2.join()
        # [myAccuracy2,myPrecision2,myF1score2,myRecall2] = result_queue2.get()
                              