
import nltk

#nltk.download()

##pré-processamento dos dados
base = [('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia está muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')]

##1)eliminar as stop-words(sem significado real para análise na frase)

stopwordnltk = nltk.corpus.stopwords.words('portuguese')

def removestopwords(texto):
        frases= []
        for (palavras,emocao) in texto:
                top = [p for p in palavras.split() if p not in stopwordnltk]
                frases.append((top,emocao))
        return frases

#print(removestopwords(base))

#2)Aplicação de steamming(radical da palavra)

def aplicacaosteammer(texto):
        steammer = nltk.stem.RSLPStemmer() #próprio para lingua portuguesa
        frasessteamming = []
        for(palavras,emocao) in texto:
                comsteming = [str(steammer.stem(p)) for p in palavras.split() if p not in stopwordnltk]
                frasessteamming.append((comsteming,emocao))
        return frasessteamming

frasessteammingg = aplicacaosteammer(base)
#print(frasessteammingg)

#3)retirar a classe/emoção vinculada a cada frase

def buscapalavras(frases):
        todaspalavras = []
        for (palavras,emocao)in frases:
                todaspalavras.extend(palavras)
        return todaspalavras
everyword = buscapalavras(frasessteammingg)
#print(everyword)

def buscafrequencia(palavra):
        palavra = nltk.FreqDist(palavra)
        return palavra

frequencia = buscafrequencia(everyword)
#print(frequencia.most_common(50))

def buscapalavrasunicas(frequencia):
        freq = frequencia.keys()
        return freq

palavrasunicas = buscapalavrasunicas(frequencia)
#print(palavrasunicas)

def extratorpalavras(documento):
        doc = set(documento)
        carac = { }
        for palavras in palavrasunicas:
                carac['%s' %palavras] = (palavras in doc)
        return carac

basecompleta = nltk.classify.apply_features(extratorpalavras,frasessteammingg)
#print(basecompleta[0])

#constrói a tabela de probabilidade e treina
classificador = nltk.NaiveBayesClassifier.train(basecompleta)
#checar se os classificadores estão certos

#print(classificador.labels())

#mostra a tabela de probabilidade montada
#print(classificador.show_most_informative_features())

teste = 'estou com medo'
#extrair o radical dessa frase que quero testar
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavras)in teste.split():
        comstem = [p for p in palavras.split()]
        testestemming.append(str(stemmer.stem(comstem[0])))

novo = extratorpalavras(testestemming)
#print(novo)

print(classificador.classify(novo))
distribuicao = classificador.prob_classify(novo)
for classe in distribuicao.samples():
        print('%s: %f'%(classe,distribuicao.prob(classe)))
