import pickle
import anc2vec.train as builder
embeds = builder.fit('go.obo', embedding_sz=200, batch_sz=64, num_epochs=1000)

with open('embeds.pkl', 'wb') as fp:
    pickle.dump(embeds, fp)
