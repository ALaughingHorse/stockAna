def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def get_cross_overs(short,long):
    """
    Get cross-ovrers from two curves
    
    ---Parameters
    
    short: the curve that fluctuates more. i.e. a smoothed curve with lower window
    
    long: the curve that fluctuates less. i.e. a smoothed curve with longer window
    
    ---Returns
    
    dictionary containing buy or sell index calculated with the crossovers from the curve
    
    
    """

    assert(len(short) == len(long))    
    
    buy_idx = [i for i in list(range(1,len(short))) if ((short[i-1] < long[i-1]) & (short[i] > long[i]))]
    
    sell_idx = [i for i in list(range(1,len(short))) if ((short[i-1] > long[i-1]) & (short[i] < long[i]))]
    
    return {'buy':buy_idx,'sell':sell_idx}


def plot_trade_sig(vec,short,long):
    
    """
    Plot the trading signals
    
    ---Paramters
    
    vec: the stock price
    
    short/long: same as described in the function 'get_corss_overs'
    
    ---Returns
    dictionary containing buy or sell index calculated with the crossovers from the curve
    """
    
    co  = get_cross_overs(short,long)
    
    normalizedVec = (np.array(vec) - min(vec))/(max(vec) - min(vec))
    
    vec.plot(figsize = Psize,title = 'Trading signals (red for buy, green for sales)')
    
    for idx in co['sell']:
        plt.axvline(idx,ymin = normalizedVec[idx] - 0.1, color = 'g',\
                ymax = normalizedVec[idx] + 0.1)
    
    for idx in co['buy']:
        plt.axvline(idx,ymin = normalizedVec[idx] - 0.1, color = 'r',\
                ymax = normalizedVec[idx] + 0.1)
        
    return co

def get_performance(price, buy_idx, sell_idx):
    
    """
    Get the performance of a set of buying and selling signals
    
    ---Parameter
    
    price: the price of a certain stock where the buying and selling signals are derived from
    
    buy_idx: the buying index
    
    sell_idx: the selling index
    
    ---Return
    
    all_gains: percentage gains from each buy and sell pairs
    
    all_loss_prevent: percentage loss prevented from each sell and buy pairs
    
    roi: Return of investment in decimal percentage
    
    tot_lossPrev: Percentage of total loss prevented from sell signals
    
    messge: a summarizing message of the performance
    """
    
    import numpy as np
    #' First make sure the buy and sell signals are of the same length, starting with a buy signal and ending with a sell signal
    
    if min(buy_idx) > min(sell_idx):
        sell_idx = sell_idx[1:]
        
    if max(buy_idx) > max(sell_idx):
        buy_idx = buy_idx[:-1]
        
    assert(len(buy_idx) == len(sell_idx))
    
    # Calculate buy gains
    buy_price = price[buy_idx]
    sell_price = price[sell_idx]
    
    from functools import reduce
    percent_gains = (np.array(sell_price) - np.array(buy_price))/np.array(buy_price)
    total_gain = reduce(lambda x, y: x*y, (percent_gains + 1))
    
    roi = total_gain - 1
    
    # Calculate sell loss prevention
    buy_price_sub = buy_price[1:]
    sell_price_sub = sell_price[:-1]
    
    percent_loss_prevent = (np.array(sell_price_sub) - np.array(buy_price_sub))/np.array(sell_price_sub)
    total_loss_prevents =  reduce(lambda x, y: x*y, (1 + percent_loss_prevent))
    
    msg = "The natural ROI throughout the period is {}, \
    total ROI if following every buy and sell signal is {}. Total loss prevention is {}.\
    {} of buys are successful while {} of sells are successful. Maximum percentage gain per trade is {} while maximum percentage loss per trade is {}".format((list(price)[-1]-price[0])/price[0], roi, total_loss_prevents, sum(percent_gains > 0)/len(buy_idx),
                                                                        sum(percent_loss_prevent > 0)/len(sell_idx), max(percent_gains),min(percent_gains))
    
    print(msg)
    
    class out:
        
        all_gains = percent_gains
        all_loss_prevent = percent_loss_prevent
        
        roi = roi
        tot_lossPrev = total_loss_prevents
        
        message = msg
    
    return out

def get_confidence_info(price, buy_idx, sell_idx):
    
    """
    Get information about how confidence should we be for each buy signal
    
    ---Parameters:
    
    same as get_performance function
    
    ---Return:
    
    gains: a vector of gains for each buy and sell combination
    
    min_since_purchase: minimum price since the most recent purchase, before the next sell
    
    percent_loss_since_purchase: percentage of the min_since_purchase
    
    day_since_last_sells: number of days between a buy signal and the sell signal before it
    """
    
    import numpy as np
    #' First make sure the buy and sell signals are of the same length, starting with a buy signal and ending with a sell signal
    
    if min(buy_idx) > min(sell_idx):
        sell_idx = sell_idx[1:]
        
    if max(buy_idx) > max(sell_idx):
        buy_idx = buy_idx[:-1]
        
    assert(len(buy_idx) == len(sell_idx))
    
    # Calculate buy gains
    buy_price = price[buy_idx]
    sell_price = price[sell_idx]
    percent_gains = (np.array(sell_price) - np.array(buy_price))/np.array(buy_price)
    
    min_since_buy = []
    day_since_last_sell = []
    
    for i in list(range(len(buy_idx))):
        
        price_vec = price[(buy_idx[i] + 1):sell_idx[i]]
        
        if len(price_vec) == 0:
            min_since_buy.append(min([price[buy_idx[i]],price[sell_idx[i]]]))
        else:
            min_since_buy.append(min(price_vec))
        
        if i == 0:
            day_since_last_sell.append(-1)
        else:
            day_since_last_sell.append(buy_idx[i] - sell_idx[i-1])
            
    percent_loss_since_buy = (np.array(min_since_buy) - np.array(price[buy_idx]))/np.array(price[buy_idx])
    
    class out:
        
        gains = percent_gains
        
        min_since_purchase = min_since_buy
        
        percent_loss_since_purchase = percent_loss_since_buy
        
        day_since_last_sells = day_since_last_sell
    
    return out

def get_rich(price_full,day_num = 'max'):
    
    """
    One function to get all summaries around MACD/Signal line crossing signals
    """
    
    if day_num == 'max':
        price = price_full
    else:
        price = price_full[-day_num:].reset_index(drop = True)
    #Calculate the exponential moving average
    
    price_ema_12 = pd.ewma(price,span = 12,adjust = False)
    price_ema_26 = pd.ewma(price,span = 26, adjust = False)
    
    #Calculate MACD and signal line
    macd = np.array(price_ema_12) - np.array(price_ema_26)
    signal = pd.ewma(macd,span = 9, adjust = False)
    
    # Get buy/sell indexes
    idxes = get_cross_overs(short = macd, long = signal)
    buy = idxes['buy']
    sell = idxes['sell']
    
    # Plot trade signals
    plot_trade_sig(price,macd,signal)
    
    # Get performances
    perf = get_performance(price = price, buy_idx = buy, sell_idx = sell)
    
    # Get confidence info
    conf = get_confidence_info(price = price, buy_idx = buy, sell_idx = sell)
    
    # Construct data frame with confidence info
    ps = pd.DataFrame({"gain":conf.gains,"day_since_sell":conf.day_since_last_sells,"max_loss":conf.percent_loss_since_purchase})
    ps = ps.loc[ps['day_since_sell'] > 0].reset_index(drop = True)
    ps['gain_flag'] = (ps['gain'] > 0).astype(int)
    
    # Get the decision tree
    from sklearn import tree
    import graphviz
    cart = tree.DecisionTreeClassifier(max_depth=2)
    cart_full = cart.fit(ps[['day_since_sell','max_loss']],ps['gain_flag'])
    
    cart_plot = tree.export_graphviz(cart_full, out_file=None,feature_names = ['day_since_sell','max_loss']) 
    graph = graphviz.Source(cart_plot)
    
    class out:
        
        calculated_macd = macd
        calculated_signal = signal
        buy_idxes = buy
        sell_idxes = sell
        performance = perf
        confidence_info = conf
        confidence_table = ps
        cart_graph = graph
    
    return out
    