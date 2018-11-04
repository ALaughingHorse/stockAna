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


def plot_trade_sig(vec,short,long,max_day = 'max',Psize = (30,20)):
    
    """
    Plot the trading signals
    
    ---Paramters
    
    vec: the stock price
    
    short/long: same as described in the function 'get_corss_overs'
    
    max_day: how manys days of data to display on plot
    
    ---Returns
    dictionary containing buy or sell index calculated with the crossovers from the curve
    """
    import numpy as np
    from matplotlib import pyplot as plt

    if max_day == 'max':
        max_day = len(vec)
    
    co  = get_cross_overs(short,long)
     
    vec[-max_day:].plot(figsize = Psize)
    plt.plot(co['buy'],vec[co['buy']].values,'o',color = 'Red',markersize = 8)
    plt.plot(co['sell'],vec[co['sell']].values,'o',color = 'Green',markersize = 8)
    plt.xlabel("index",fontsize=18)
    plt.ylabel('price',fontsize=16)
    plt.suptitle('Trading signals (red for buy, green for sales)',fontsize = 20)
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
    
    
    class out:
        
        all_gains = percent_gains
        all_loss_prevent = percent_loss_prevent
        
        roi_num = roi
        tot_lossPrev = total_loss_prevents
        
        message = msg
    
    return out

def get_confidence_info(price, price_low, buy_idx, sell_idx):
    
    """
    Get information about how confidence should we be for each buy signal
    
    ---Parameters:
    
    same as get_performance function
    
    price_low: lowest price at everyday window
    
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
    low_since_buy = []
    day_since_last_sell = []
    
    for i in list(range(len(buy_idx))):
        
        price_vec_low = price_low[(buy_idx[i] + 1):sell_idx[i]]
        price_vec = price[(buy_idx[i] + 1):sell_idx[i]]
        
        if len(price_vec_low) == 0:
            low_since_buy.append(min([price_low[buy_idx[i]],price_low[sell_idx[i]]]))
            min_since_buy.append(min([price[buy_idx[i]],price[sell_idx[i]]]))
        else:
            low_since_buy.append(min(price_vec_low))
            min_since_buy.append(min(price_vec))
        
        
        if i == 0:
            day_since_last_sell.append(-1)
        else:
            day_since_last_sell.append(buy_idx[i] - sell_idx[i-1])
            
    max_percent_loss_since_buy = (np.array(low_since_buy) - np.array(price[buy_idx]))/np.array(price[buy_idx])
    daily_percent_loss_since_buy = (np.array(min_since_buy) - np.array(price[buy_idx]))/np.array(price[buy_idx])
    
    class out:
        
        gains = percent_gains
        
        min_since_purchase = min_since_buy
        
        low_since_purchse = low_since_buy
        
        daily_percent_loss_since_purchase = daily_percent_loss_since_buy
        
        max_percent_loss_since_purchase = max_percent_loss_since_buy
        
        day_since_last_sells = day_since_last_sell
    
    return out


def get_macd_signal(price):
    """
    A function to calculate the macd and signal line
    """
    import pandas as pd
    import numpy as np
    
    price = pd.Series(price)
    price_ema_12 = price.ewm(span = 12,adjust = False).mean()
    price_ema_26 = price.ewm(span = 26, adjust = False).mean()
    
    #Calculate MACD and signal line
    macd = np.array(price_ema_12) - np.array(price_ema_26)
    signal = pd.Series(macd).ewm(span = 9, adjust = False).mean()
    
    return({'macd':macd,'signal':signal})

def get_rich(price,price_low,day_num = 'max',Psize = (30,20)):
    
    """
    One function to get all summaries around MACD/Signal line crossing signals
    """
    import pandas as pd
    import numpy as np
   
    price  = pd.Series(price)
    
    # get macd and signal lines
    macd_signal = get_macd_signal(price)
    
    macd = macd_signal['macd']
    signal = macd_signal['signal']
    
    # Create a version of list so that we can select the last element more conveniently
    macd_l = list(macd)
    signal_l = list(signal)
    
    
    if((macd_l[-2] < signal_l[-2]) & (macd_l[-1] >= signal_l[-1])):
        instruction_msg = 'Buy'
        print('Buy')
    if((macd_l[-2] > signal_l[-2]) & (macd_l[-1] <= signal_l[-1])):
        instruction_msg = 'Sell'
        print('Sell')
    if((macd_l[-2] < signal_l[-2]) == (macd_l[-1] < signal_l[-1])):
        print('Hold')
        latest = list(price)[-1]
        macd_temp = macd_l
        signal_temp = signal_l

        # When the latest signal is a buy signal 
        if(macd_l[-1] >= signal_l[-1]):
            action = 'sell'
            # Get a fake tomorrow price
            price_delta_percent = -0.01
            tmr_price = latest * (1+price_delta_percent)

            # With the fake tomorrow price, calculate fake macd and signal lines
            price_temp = pd.Series(np.append(price.values,[tmr_price]))
            macd_signal_temp = get_macd_signal(price_temp)
            macd_temp = list(macd_signal_temp['macd'])
            signal_temp = list(macd_signal_temp['signal'])

            # While the fake tomorrw price is not low enough to generate a sell signal, repeat the calculation above
            while macd_temp[-1] > signal_temp[-1]:

                tmr_price = tmr_price * (1+price_delta_percent)
                price_temp = pd.Series(np.append(price.values,[tmr_price]))
                macd_signal_temp = get_macd_signal(price_temp)
                macd_temp = list(macd_signal_temp['macd'])
                signal_temp = list(macd_signal_temp['signal'])

        else: # When the latest signal is a sell signal
            action = 'buy'
            # Get a fake tomorrow price
            price_delta_percent = 0.01
            tmr_price = latest * (1+price_delta_percent)

            # With the fake tomorrow price, calculate fake macd and signal lines
            price_temp = pd.Series(np.append(price.values,[tmr_price]))
            macd_signal_temp = get_macd_signal(price_temp)
            macd_temp = list(macd_signal_temp['macd'])
            signal_temp = list(macd_signal_temp['signal'])

            # While the fake tomorrw price is not low enough to generate a sell signal, repeat the calculation above
            while macd_temp[-1] <= signal_temp[-1]:

                tmr_price = tmr_price * (1+price_delta_percent)
                price_temp = pd.Series(np.append(price.values,[tmr_price]))
                macd_signal_temp = get_macd_signal(price_temp)
                macd_temp = list(macd_signal_temp['macd'])
                signal_temp = list(macd_signal_temp['signal'])


        instruction_msg = "If the closing price tomorrow reaches {}, {} this stock".format(tmr_price,action)

    
    # Get buy/sell indexes
    idxes = get_cross_overs(short = macd, long = signal)
    buy = idxes['buy']
    sell = idxes['sell']
    
    # Plot trade signals
    plot_trade_sig(price,macd,signal,max_day = day_num)
    
    # Get performances
    perf = get_performance(price = price, buy_idx = buy, sell_idx = sell)
    
    # Get confidence info
    conf = get_confidence_info(price = price, price_low = price_low, buy_idx = buy, sell_idx = sell)
    
    # Construct data frame with confidence info
    ps = pd.DataFrame({"gain":conf.gains,"day_since_sell":conf.day_since_last_sells,"daily_loss":conf.daily_percent_loss_since_purchase,"max_loss":conf.max_percent_loss_since_purchase})
    ps = ps.loc[ps['day_since_sell'] > 0].reset_index(drop = True)
    ps['gain_flag'] = (ps['gain'] > 0).astype(int)
    
    # Get the decision tree
    from sklearn import tree
    import graphviz
    cart = tree.DecisionTreeClassifier(max_depth=3)
    cart_full = cart.fit(ps[['day_since_sell','daily_loss']],ps['gain_flag'])
    
    cart_plot = tree.export_graphviz(cart_full, out_file=None,feature_names = ['day_since_sell','daily_loss']) 
    graph = graphviz.Source(cart_plot)
    
    class out:
        
        calculated_macd = macd
        calculated_signal = signal
        buy_idxes = buy
        sell_idxes = sell
        performance = perf
        confidence_info = conf
        confidence_table = ps
        cart_model = cart_full

        cart_graph = graph
        instruction = instruction_msg
    
    return out

def run_strategy(close,strategy_func):

    # Make sure we have enough data
    import numpy as np
    assert(len(close) >=7)

    # Initialize the variables
    all_action = []
    position = 'waiting'
    simple_return = 0
    simple_returns = []
    total_portfolio = 1
    portfolio_hist = []

    bought_at = -1
    sold_at = -1

    for i in range(7,len(close)):

        action  = strategy_func(close[i],close[:i])
        all_action = all_action + [action]

        if action == 'buy':
            position = 'holding'
            bought_at =  close[i]

        if (position == 'holding') & (action == 'sell'):
            position = 'waiting'
            sold_at = close[i]
            simple_return = (sold_at - bought_at)/bought_at

        elif action == 'hold':
            simple_return = 0

        simple_returns = simple_returns + [simple_return]
        total_portfolio = total_portfolio * (1 + simple_return)

        portfolio_hist = portfolio_hist + [total_portfolio]


    simple_returns = [0] * 7 + simple_returns
    portfolio_hist = [1] * 7 + portfolio_hist

    buy_idx = [x for x in list(range(len(all_action))) if all_action[x] == 'buy']
    sell_idx = [x for x in list(range(len(all_action))) if all_action[x] == 'sell']


    plt.subplot(311)
    ax1 = plt.subplot(311)
    close.plot(figsize = Psize)
    plt.plot(buy_idx,close[buy_idx].values,'o',color = 'Green',markersize = 4)
    plt.plot(sell_idx,close[sell_idx].values,'o',color = 'Red',markersize = 4)
    plt.ylabel('Price')
    plt.title('Trading Signals (green for buy, red for sell)')

    plt.subplot(312,sharex=ax1)
    plt.plot(simple_returns)
    plt.title('Simple Returns')

    plt.subplot(313,sharex=ax1)
    plt.plot(portfolio_hist)
    plt.title('Total Portfolio')
    
    msg = "The natural ROI throughout the period is {}, \
    total ROI if following every buy and sell signal is {}".format((close[len(close)-1]-close[0])/close[0],total_portfolio - 1)
    
    print(msg)