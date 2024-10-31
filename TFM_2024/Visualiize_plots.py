from MODEL.Optimitation_and_plotting import *



def generate_all_plots(losses_df, time_df):

    print(losses_df)
    data, mean_by_epoch, mean_by_node = prepare_data(losses_df)

    plot_loss_by_epoch(mean_by_epoch)
    plot_accuracy_by_epoch(mean_by_epoch)
    plot_loss_heatmap(mean_by_node)
    plot_performance_by_node(mean_by_node, metric = 'val_loss')
    plot_performance_by_node(mean_by_node, metric = 'accuracy', output_file = 'accuracy_by_node.png')
    plot_training_time(time_df)


if __name__ == "__main__":

    losses = pd.read_csv (METRICS_FILE, sep = ',', index_col = 0)
    times =pd.read_csv (TIME_FILE, sep = ',', index_col = 0)
    generate_all_plots(losses,time)