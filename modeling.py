# from matplotlib.widgets import TextBox
# from matplotlib.pyplot import pause, xticks
# from numpy.core.shape_base import block
# from IPython.display import display
# from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# from pandas.core.indexing import convert_from_missing_indexer_tuple
# import matplotlib.gridspec as gridspec
import seaborn as sns
import itertools
from cycler import cycler
# from seaborn.external.husl import lab_e
from sklearn import decomposition, cross_decomposition
from sklearn.metrics import (
    mean_squared_error, classification_report, roc_auc_score, roc_curve)
from sklearn import model_selection
import dffuncs as dff
import sys
sys.path.append('/home/lepa/Documents/python/')


class PCA:
    """Apply principal component analysis to the given dataframe."""

    def __init__(
        self,
        df,
        n_components=10,
        transpose_df=False,
        col_as_header=None,
        summary=True,
        hue=None
    ):
        """
        """
        self.df = df
        self.n_components = n_components
        self.hue = hue

        if transpose_df:
            self.df = dff.transpose(self.df, col_as_header=col_as_header)

        # apply pca
        self.pca, self.pca_transformed = self._apply_PCA(self.n_components,
                                                         self.df)

        # get some useful quantities
        self.n_pcs_ = self.pca.n_components_
        self.exp_var_ = self.pca.explained_variance_
        self.exp_var_rat_ = self.pca.explained_variance_ratio_

        self.variance_df_ = self._variance()
        self.scores_df_ = self._scores()
        self.loadings_df_ = self._loadings()

        if summary:
            self.summary_plot()

    def _apply_PCA(self, n_components, df):
        """Apply PCA and transform the dataframe.
        Returns a PCA object and its transformation.
        """
        pca = decomposition.PCA(n_components)

        return pca, pca.fit_transform(df)

    def _variance(self):
        """Create a dataframe containing the variance, the cumulative variance
        and their respective ratios explained by the PCA components.
        """
        self.cum_exp_var_ = np.cumsum(self.exp_var_)
        self.cum_exp_var_rat_ = np.cumsum(self.exp_var_rat_)

        variances = {
            'Explained variance': self.exp_var_,
            'Cumulative explained variance': self.cum_exp_var_,
            'Explained variance ratio': self.exp_var_rat_,
            'Cumulative explained variance ratio': self.cum_exp_var_rat_
        }

        variance_df = pd.DataFrame(
            variances,
            index=[f'PC{i + 1}' for i in range(self.n_pcs_)],
            columns=variances.keys()
        ).T

        return variance_df

    def _scores(self):
        """Create a dataframe containing the PCA scores for each component.
        """
        scores_df = pd.DataFrame(
            self.pca_transformed,
            index=self.df.index,
            columns=[f'PC{i + 1}' for i in range(self.n_pcs_)]
        )

        return scores_df

    def _loadings(self):
        """Create a dataframe containing the PCA loadings.
        """
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            index=self.df.columns,
            columns=[f'PC{i + 1}' for i in range(self.n_pcs_)]
        )

        return loadings_df

    # add n_pcs????
    def scree_plot(
        self,
        figsize=None,
        show='all',
        color=None,
        marker=['o', 's'],
        markersize=None,
        label=None,
        title=None,
        title_y=1,
        xlabel='Principal components',
        ylabel='Explained variance ratio (%)',
        ylim=[-0.05, 1.05],
        font_scale=1,
        legend=True,
        legend_facecolor=None,
        legend_alpha=None,
        legend_pos='center right',
        label_points=True,
        label_points_offset=0.02,
        **kwargs
    ):
        """Scree plot of the PCA components.
        """
        # funtion to help adding text
        def add_text(ax, xys_list, fontsize=10):
            """Add text to the 'ax' object.
            Parameter 'xys' is a list/tuple containing lists/tuples of
            the form (x, y, s).
            'x','y' are coordinates that define the position of the text.
            's' is a the text.
            """
            for i in xys_list:
                x, y, s = i
                ax.text(x, y, s, ha='center', va='bottom',
                        # fontsize=fontsize
                        )

        # select elements to plot and add the text to add above each point
        if show == 'all':
            to_plot = ['Explained variance ratio',
                       'Cumulative explained variance ratio']
            text = list(zip(
                range(self.variance_df_.shape[1]),
                self.variance_df_.iloc[2] + label_points_offset,
                self.variance_df_.iloc[2].multiply(100).map(
                    '{:.2f}'.format)
            ))
            # index starts at 1 instead of 0 in order to avoid adding the
            # text for the first element twice
            text.extend(
                list(zip(
                    range(1, self.variance_df_.shape[1]),
                    self.variance_df_.iloc[3, 1:] + label_points_offset,
                    self.variance_df_.iloc[3, 1:].multiply(100).map(
                        '{:.2f}'.format)
                ))
            )
        elif show == 'exp_var_rat':
            to_plot = 'Explained variance ratio'
            text = list(zip(
                range(self.variance_df_.shape[1]),
                self.variance_df_.iloc[2] + label_points_offset,
                self.variance_df_.iloc[2].multiply(100).map(
                    '{:.2f}'.format)
            ))
        elif show == 'cum_exp_var_rat':
            to_plot = 'Cumulative explained variance ratio'
            text = list(zip(
                range(self.variance_df_.shape[1]),
                self.variance_df_.iloc[3] + label_points_offset,
                self.variance_df_.iloc[3].multiply(100).map(
                    '{:.2f}'.format)
            ))

        with sns.plotting_context('notebook', font_scale=font_scale):
            # plot
            ax = self.variance_df_.T.plot(
                y=to_plot,
                figsize=figsize,
                # title=title,
                color=color,
                ylim=ylim,
                markersize=markersize,
                legend=False,
                **kwargs
            )

            # set x and y labels
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # set the desired markers and labels for each line
            for i, line in enumerate(ax.get_lines()):
                if label is not None:
                    line.set_label(label[i])
                if marker is not None:
                    line.set_marker(marker[i])

            if legend:
                # set the correct lines and symbols to the legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, facecolor=legend_facecolor,
                          framealpha=legend_alpha,
                          loc=legend_pos)

            # set text above each point
            if label_points:
                add_text(ax, text)

            # get the figure of the current axes object to return it for
            # convenience after plotting and set title
            fig = ax.get_figure()
            fig.suptitle(title, y=title_y)

        return fig, ax

    # should change {x, y}_vars name to {x, y}_pcs

    # pass self.hue instead of hue in all functions below
    def scores_pairplot(
        self,
        n_pcs=None,
        x_vars=None,
        y_vars=None,
        # show_percent=False,
        # hue=None,
        hue_order=None,
        color=None,
        alpha=1,
        markers=None,
        markersize=30,
        markeredgecolor='white',
        height=2.5,
        aspect=1,
        title=None,
        title_y=1,
        font_scale=1,
        legend=True,
        diag_kind=None,
        plot_kws=None,
        diag_kws=None,
        grid_kws=None,
        conf_ellipse=True,
        n_std=2,
        ell_linewidth=1,
        ell_fill=True,
        ell_alpha=0.25,
        **kwargs
    ):
        """Pair plot of the PCA scores.
        Returns the figure and the PairGrid instance.

        'hue' is a list/tuple of length 2 which adds a column to be used as
        the seaborn 'hue' parameter. The first element is a string that is
        used as the column's name, while the second element is a list that is
        used as the column's values.
        """
        df = self.scores_df_.iloc[:, :n_pcs].copy()

        hue_name = None
        if self.hue is not None:
            hue_name = 'Class'
            # hue_name = self.hue[0]
            # hue_vals = self.hue[1]
            df[hue_name] = self.hue

        # add items to plot_kws dictionary
        if plot_kws is None:
            plot_kws = {}
        plot_kws.update({
            's': markersize,
            'alpha': alpha,
            'edgecolor': markeredgecolor,
        })

        # use this to set the scale off all fonts in the plot
        with sns.plotting_context('notebook', font_scale=font_scale):
            # plot
            g = sns.pairplot(
                df,
                x_vars=x_vars,
                y_vars=y_vars,
                hue=hue_name,
                hue_order=hue_order,
                palette=color,
                markers=markers,
                height=height,
                aspect=aspect,
                diag_kind=diag_kind,
                plot_kws=plot_kws,
                diag_kws=diag_kws,
                grid_kws=grid_kws,
                **kwargs
            )

            # get the figure of the current PairGrid object to return it for
            # convenience after plotting and set title
            fig = g.fig
            fig.suptitle(title, y=title_y)

        if legend is False:
            g._legend.remove()

        # error ellipses

        pcs = list(itertools.product(g.x_vars, g.y_vars))
        pcs_axs = list(el for el in zip(pcs, g.axes.flatten()))
        non_diag_pcs_axs = [el for el in pcs_axs if el[0][0] != el[0][1]]

        if conf_ellipse:

            for (y_pc, x_pc), ax in non_diag_pcs_axs:

                # set color cyclers to set the colors of the ellipses
                if color is None:
                    default_palette = plt.rcParams['axes.prop_cycle']
                    default_color_cycle = cycler(
                        colors=default_palette.by_key()['color'])
                    color_cycle = default_color_cycle()
                else:
                    custom_color_cycle = cycler(colors=color)
                    color_cycle = custom_color_cycle()

                for i in df[hue_name].unique():
                    cc = next(color_cycle)

                    if ell_fill:
                        ell_facecolor = cc['colors']
                    else:
                        ell_facecolor = "None"

                    _confidence_ellipse(
                        x=df.loc[df[hue_name] == i][x_pc],
                        y=df.loc[df[hue_name] == i][y_pc],
                        n_std=n_std,
                        ax=ax,
                        # keep these hardcoded for the time being
                        edgecolor=cc['colors'],
                        linestyle='-',
                        linewidth=ell_linewidth,
                        facecolor=ell_facecolor,
                        alpha=ell_alpha,
                    )

        return fig, g

    def scores_plot(
        self,
        x_pc=1,
        y_pc=2,
        figsize=None,
        # hue=None,
        hue_order=None,
        color=None,
        alpha=1,
        font_scale=1,
        markers=True,
        markersize=35,
        markeredgecolor='white',
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        show_percent=False,
        legend='full',
        legend_facecolor=None,
        legend_alpha=None,
        legend_pos='best',
        legend_title='Class',
        title=None,
        title_y=1,
        ax=None,
        conf_ellipse=True,
        n_std=2,
        ell_linewidth=1,
        ell_fill=True,
        ell_alpha=0.25,
        **kwargs
    ):
        """PCA scores plot.
        """
        df = self.scores_df_.copy()

        hue_name = None
        if self.hue is not None:
            hue_name = 'Class'
            # hue_name = hue[0]
            # hue_vals = hue[1]
            df[hue_name] = self.hue

        if isinstance(x_pc, int):
            x_pc = f'PC{x_pc}'
        if isinstance(y_pc, int):
            y_pc = f'PC{y_pc}'

        # if markers is None:
        #     # markers = True
        #     style= None
        # else:
        #     style = hue_name

        with sns.plotting_context('notebook', font_scale=font_scale):
            ax = sns.scatterplot(
                data=df,
                x=x_pc,
                y=y_pc,
                hue=hue_name,
                hue_order=hue_order,
                palette=color,
                alpha=alpha,
                style=hue_name,
                markers=markers,
                s=markersize,
                edgecolor=markeredgecolor,
                legend=legend,
                ax=ax,
                **kwargs
            )

            # set x and y axes labels
            if xlabel is None:
                xlabel = ax.get_xlabel()
            if ylabel is None:
                ylabel = ax.get_ylabel()
            if show_percent:
                x_percent = self.variance_df_.loc['Explained variance ratio',
                                                  x_pc]
                y_percent = self.variance_df_.loc['Explained variance ratio',
                                                  y_pc]
                xlabel = f'{xlabel} ({x_percent:.2%})'
                ylabel = f'{ylabel} ({y_percent:.2%})'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if legend:
                ax.legend(
                    facecolor=legend_facecolor,
                    framealpha=legend_alpha,
                    loc=legend_pos,
                    title=legend_title
                )

            # error ellipses
            if conf_ellipse and self.hue is not None:

                # set color cyclers to set the colors of the ellipses
                if color is None:
                    default_palette = plt.rcParams['axes.prop_cycle']
                    default_color_cycle = cycler(
                        colors=default_palette.by_key()['color'])
                    color_cycle = default_color_cycle()
                else:
                    custom_color_cycle = cycler(colors=color)
                    color_cycle = custom_color_cycle()

                for i in df[hue_name].unique():
                    cc = next(color_cycle)

                    if ell_fill:
                        ell_facecolor = cc['colors']
                    else:
                        ell_facecolor = "None"

                    _confidence_ellipse(
                        x=df.loc[df[hue_name] == i][x_pc],
                        y=df.loc[df[hue_name] == i][y_pc],
                        n_std=n_std,
                        ax=ax,
                        # keep these hardcoded for the time being
                        edgecolor=cc['colors'],
                        linestyle='-',
                        linewidth=ell_linewidth,
                        facecolor=ell_facecolor,
                        alpha=ell_alpha,
                    )

            #####

            # get the figure of the current axes object to return it for
            # convenience after plotting and set title
            fig = ax.get_figure()
            if figsize is not None:
                fig.set_size_inches(figsize)
            fig.suptitle(title, y=title_y)

        return fig, ax

    def loadings_plot(
        self,
        pc=1,
        figsize=None,
        color=None,
        show_percent=True,
        title=None,
        title_y=1,
        xlabel=None,
        ylabel=None,
        legend=True,
        legend_labels=None,
        legend_facecolor=None,
        legend_alpha=None,
        legend_pos='best',
        font_scale=1,
        **kwargs
    ):
        """Loadings plot of the PCA components. 'pc' defines the
        principal component to plot.
        """
        # to account for zero-indexing
        pc = pc - 1

        # default color palette to use goes from darker to lighter shades of
        # the default matplotlib blue color
        if color is None:
            color = sns.color_palette(
                sns.light_palette('C0',
                                  n_colors=np.round(self.n_pcs_ * 1.25),
                                  reverse=True
                                  ))

        # use this to set the scale off all fonts in the plot
        with sns.plotting_context('notebook', font_scale=font_scale):
            # plot
            ax = self.loadings_df_.iloc[:, pc].plot(
                kind='line',
                figsize=figsize,
                subplots=False,
                color=color,
                legend=legend,
                **kwargs
            )

            # set x, y labels
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)

            # set the correct lines and symbols to the legend
            handles, labels = ax.get_legend_handles_labels()
            if show_percent:
                labels = [f'{self.variance_df_.columns[pc]} ' +
                          f'({self.variance_df_.iloc[2, pc]:.2%})']
            ax.legend(handles, labels, facecolor=legend_facecolor,
                      framealpha=legend_alpha,
                      loc=legend_pos)

            # get the figure of the current axes object to return it for
            # convenience after plotting and set title
            fig = ax.get_figure()
            fig.suptitle(title, y=title_y)

        return fig, ax

    def loadings_plot_all(
        self,
        n_pcs=None,
        layout=None,
        figsize=None,
        color=None,
        title=None,
        title_y=1,
        xlabel=None,
        ylabel=None,
        legend=True,
        legend_labels=None,
        legend_facecolor=None,
        legend_alpha=None,
        legend_pos='best',
        font_scale=1,
        **kwargs
    ):
        """Loadings plot of the PCA components. 'n_pcs' defines the number of
        principal components to plot.
        """
        # default color palette to use goes from darker to lighter shades of
        # the default matplotlib blue color
        if color is None:
            color = 'tab:blue'

        # use this to set the scale off all fonts in the plot
        with sns.plotting_context('notebook', font_scale=font_scale):
            # plot
            axes = self.loadings_df_.iloc[:, :n_pcs].plot(
                figsize=figsize,
                subplots=True,
                layout=layout,
                color=color,
                legend=legend,
                **kwargs
            )

            # iterator to set legend labels
            pcs_vars = list(zip(
                self.variance_df_.columns,
                self.variance_df_.iloc[2]
            ))
            labels = [f'{i} ({j:.2%})' for i, j in pcs_vars]
            iter_labels = iter(labels)

            for ax in axes.flatten():
                # set x, y labels
                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                if ylabel is not None:
                    ax.set_ylabel(ylabel)

                # set the correct lines and symbols to the legend
                handles, _ = ax.get_legend_handles_labels()
                # apparently the iterator needs to be inside the square
                # brackets or else only the first letter of the label will be
                # displayed in the legend. also see this:
                # https://stackoverflow.com/questions/10557614/matplotlib-figlegend-only-printing-first-letter
                ax.legend(
                    handles,
                    [next(iter_labels)],
                    facecolor=legend_facecolor,
                    framealpha=legend_alpha,
                    loc=legend_pos
                )

            # get the figure of the current axes object to return it for
            # convenience after plotting and set title
            fig = axes.flatten()[0].get_figure()
            fig.suptitle(title, y=title_y)

        return fig, axes

    def summary_plot(self):
        """A plot that summarizes the PCA results.
        """
        fig = plt.figure(figsize=(15, 10), constrained_layout=False)

        gs = fig.add_gridspec(5, 5)

        scree_ax = fig.add_subplot(gs[:2, 1:4])

        loadings_axes = []
        for row in range(2, 5):
            ax = fig.add_subplot(gs[row:row + 1, :2])
            loadings_axes.append(ax)

        scores_axes = []
        for row in range(2, 5):
            for col in range(2, 5):
                ax = fig.add_subplot(gs[row:row + 1, col:col + 1])
                scores_axes.append(ax)

        # split the scores_axes list to 3 sublists of length 3
        scores_axes = [scores_axes[x:x + 3]
                       for x in range(0, len(scores_axes), 3)]

        # scree plot
        scree_ax.set_title('Scree plot')
        self.scree_plot(
            ax=scree_ax,
            xlabel='',
        )

        # first 3 loadings plot
        loadings_axes[0].set_title('Loadings plots (first 3)')
        for i, ax in enumerate(loadings_axes):
            self.loadings_plot(pc=i + 1, ax=ax)
            if i != 2:
                ax.set_xlabel('')
                # ax.set_xticklabels('')

        # first 3 scores (pair)plot
        scores_axes[0][1].set_title('Scores plots (first 3)')

        xlims = []

        for i in range(3):
            for j in range(3):
                if i == j:
                    sns.kdeplot(
                        ax=scores_axes[j][i],
                        data=self.scores_df_,
                        x=self.scores_df_.columns[i],
                        hue=self.hue,
                        fill=True,
                        legend=False
                    )
                    xlims.append(scores_axes[i][j].get_xlim())
                else:
                    self.scores_plot(
                        ax=scores_axes[j][i],
                        x_pc=i + 1,
                        y_pc=j + 1,
                        # hue=self.hue,
                        legend='full',
                        markersize=30,
                        markeredgecolor='white',
                        # legend_pos=(1, 0.5),
                        # legend_alpha=0,
                        show_percent=True
                    )

                # remove all legends except right-most middle plot's
                # if i != 2 or j != 1:
                    scores_axes[j][i].legend().set_visible(False)

        # set common xlims for scores plots of the same pc
        for i in range(3):
            for j in range(3):
                scores_axes[j][i].set_xlim(xlims[i])

        pc1_percent = self.variance_df_.loc['Explained variance ratio', 'PC1']
        pc3_percent = self.variance_df_.loc['Explained variance ratio', 'PC3']
        scores_axes[0][0].set_ylabel(f'PC1 ({pc1_percent:.2%})')
        scores_axes[2][2].set_xlabel(f'PC3 ({pc3_percent:.2%})')

        # keep tick labels and axes labels for the left-most and bottom-most
        # axes only
        for i in range(2):
            for ax in scores_axes[i]:
                ax.set_xlabel(None)
                # ax.get_xaxis().set_ticklabels([])
            for j in range(3):
                scores_axes[j][i + 1].set_ylabel(None)
                # scores_axes[j][i + 1].get_yaxis().set_ticklabels([])

        handles, labels = scores_axes[1][0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            title='Class',
            loc='center left',
            bbox_to_anchor=(1.0, 0.32),
            facecolor='white',
            edgecolor='white'
        )

        # fig.subplots_adjust(
        #     top=1.5,
        #     bottom=0.4
        #     )
        # fig.subplots_adjust(wspace=0.3, hspace=0.1)

        fig.set_tight_layout(True)

        return fig


class PLS:
    """Apply partial least squares regression to the given data.
    """

    def __init__(
        self,
        x_train,
        x_test,
        y_train,
        y_test,
        n_components=2,
        max_iter=500,
        scale=False,
        tol=1e-06,
        round_prediction=False,
        cv_n_components=15,
        cv=5
    ):
        """
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        # self.df = df
        # self.target = target
        self.n_components = n_components
        self._max_iter = max_iter
        self._scale = scale
        self._tol = tol
        self._round_prediction = round_prediction
        # self._random_state = random_state
        self.cv_n_components = cv_n_components
        # Determines the cross-validation splitting strategy.
        # See sklearn documentation
        self.cv = cv

        # if transpose_df:
        #     self.df = dff.df_transpose(self.df, col_as_header=col_as_header)

        # # split dataset
        # (self.x_train, self.x_test,
        #     self.y_train, self.y_test) = model_selection.train_test_split(
        #         self.df,
        #         self.target,
        #         test_size=test_size,
        #         random_state=random_state,
        #         shuffle=shuffle,
        #         # maybe make stratify a boolean, with target as its argument
        #         stratify=stratify
        # )

        if self.n_components == 'auto':
            self.n_components = self.rmsecv_min()
            print(f'Using {self.n_components} components.')
        elif self.n_components == 'ask':
            self.rmsecv_plot()
            # show plot before entering number of components
            plt.show()
            plt.gcf().canvas.draw()

            self.n_components = int(input('Input number of components:'))

        # apply pls
        self.pls, self.pls_fit = self._apply_PLS(
            self.x_train,
            self.y_train,
            self.n_components,
            max_iter=self._max_iter,
            scale=self._scale,
            tol=self._tol
        )

        self.prediction_df_ = self.predict(
            # round_to_int=self._round_prediction
        )

        self.x_scores_df_ = self._x_scores()
        self.y_scores_df_ = self._y_scores()
        self.coef_df_ = self._coef()

    def _apply_PLS(self, x_train, y_train, n_components,
                   max_iter=500, scale=True, tol=1e-06):
        """Apply the PLS algorithm.
        """
        pls = cross_decomposition.PLSRegression(
            n_components=n_components,
            max_iter=max_iter,
            scale=scale,
            tol=tol
        )

        pls_fit = pls.fit(x_train, y_train)

        return pls, pls_fit

    def _x_scores(self):
        """Create a dataframe containing the x_scores of the PLS object.
        """
        x_scores_df_ = pd.DataFrame(
            self.pls.x_scores_,
            index=self.x_train.index,
            columns=[f'LV{i + 1}' for i in range(self.pls.n_components)]

        )

        return x_scores_df_

    def _y_scores(self):
        """Create a dataframe containing the x_scores of the PLS object.
        """
        y_scores_df_ = pd.DataFrame(
            self.pls.y_scores_,
            index=self.x_train.index,
            columns=[f'LV{i + 1}' for i in range(self.pls.n_components)]

        )

        return y_scores_df_

    def _coef(self):
        """Create a dataframe containing the coefficients of the PLS object.
        """
        coef_df_ = pd.DataFrame(
            self.pls.coef_,
            columns=['Coefficients'],
            index=self.x_train.columns
        )

        return coef_df_

    def predict(self):
        """Apply the dimension reduction learned on the train data and make a
        prediction for the test data.
        """
        pred_df = pd.DataFrame(
            self.y_test,
            columns=["Target"]
        )

        y_pred = self.pls.predict(self.x_test)
        pred_df['Prediction'] = y_pred

        if self._round_prediction:
            pred_df['Rounded_prediction'] = pred_df['Prediction'].round(
                decimals=0).astype(int)
            # set -0 to 0
            pred_df.loc[pred_df['Rounded_prediction'] == -0,
                        'Rounded_prediction'] = 0

        # set the sample names as index
        pred_df.index = self.x_test.index

        return pred_df

    def prediction_plot(
        self,
        y=None,
        round_to_int=False,
        figsize=None,
        linestyle=['-', '--', ':'],
        color=None,
        alpha=None,
        marker=['s', 'o', 'd'],
        markersize=None,
        xlabel='Samples',
        ylabel=None,
        label=None,
        title=None,
        title_y=1,
        font_scale=1,
        show_xticks=False,
        xticks_rotation=45,
        xticks_ha='right',
        legend=True,
        legend_facecolor=None,
        legend_alpha=None,
        legend_pos='best',
        **kwargs
    ):
        """Plot the predicted vs target data.
        """
        # use this to set the scale off all fonts in the plot
        with sns.plotting_context('notebook', font_scale=font_scale):
            ax = self.prediction_df_.plot(
                kind='line',
                y=y,
                figsize=figsize,
                # linestyle=linestyle,
                color=color,
                alpha=alpha,
                style=linestyle,
                markersize=markersize,
                legend=False,
                **kwargs
            )

        # add x ticks
        if show_xticks:
            ax.set_xticks(list(range(len(self.y_test))))
            ax.set_xticklabels(self.x_test.index, rotation=xticks_rotation,
                               ha=xticks_ha)
        else:
            ax.set_xticks([])

        # set x and y labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # set the desired markers and labels for each line
        for i, line in enumerate(ax.get_lines()):
            if label is not None:
                line.set_label(label[i])
            if marker is not None:
                line.set_marker(marker[i])

        if legend:
            # set the correct lines and symbols to the legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, facecolor=legend_facecolor,
                      framealpha=legend_alpha,
                      loc=legend_pos)

        fig = ax.get_figure()
        fig.suptitle(title, y=title_y)

        return fig, ax

    def x_scores_pairplot(
        self,
        n_lvs=None,
        x_vars=None,
        y_vars=None,
        # show_percent=False,
        # hue=None,
        hue_order=None,
        color=None,
        alpha=1,
        markers=None,
        markersize=30,
        markeredgecolor='white',
        height=2.5,
        aspect=1,
        title=None,
        title_y=1,
        font_scale=1,
        legend=True,
        legend_title=None,
        target_labels=None,
        diag_kind=None,
        plot_kws=None,
        diag_kws=None,
        grid_kws=None,
        conf_ellipse=True,
        n_std=2,
        ell_linewidth=1,
        ell_fill=True,
        ell_alpha=0.25,
        **kwargs
    ):
        """Discriminant analysis pairplot using PLS.
        """
        # pls x_scores dataframe
        x_scores_df_ = self._x_scores().iloc[:, :n_lvs]

        # target labels is a dictionary that maps the integer target values
        # (0, 1, ...) to teh specified string values
        if target_labels is None:
            target = self.y_train
        else:
            target = list(map(target_labels.get, self.y_train))

        if legend_title is None:
            x_scores_df_['Target'] = target
            hue = 'Target'
        else:
            x_scores_df_[legend_title] = target
            hue = legend_title

        # add items to plot_kws dictionary
        if plot_kws is None:
            plot_kws = {}
        plot_kws.update({
            's': markersize,
            'alpha': alpha,
            'edgecolor': markeredgecolor,
        })

        # use this to set the scale off all fonts in the plot
        with sns.plotting_context('notebook', font_scale=font_scale):
            g = sns.pairplot(
                x_scores_df_,
                x_vars=x_vars,
                y_vars=y_vars,
                hue=hue,
                hue_order=hue_order,
                palette=color,
                markers=markers,
                height=height,
                aspect=aspect,
                diag_kind=diag_kind,
                plot_kws=plot_kws,
                diag_kws=diag_kws,
                grid_kws=grid_kws,
                **kwargs
            )

            # get the figure of the current PairGrid object to return it for
            # convenience after plotting and set title
            fig = g.fig
            fig.suptitle(title, y=title_y)

            plt.setp(g._legend.get_title(), fontsize=12*font_scale)
            g._legend.set_bbox_to_anchor((1., 0.5))

            if legend is False:
                g._legend.remove()

            # error ellipses

            lvs = list(itertools.product(g.x_vars, g.y_vars))
            lvs_axs = list(el for el in zip(lvs, g.axes.flatten()))
            non_diag_lvs_axs = [el for el in lvs_axs if el[0][0] != el[0][1]]

            if conf_ellipse:

                for (y_lv, x_lv), ax in non_diag_lvs_axs:

                    # set color cyclers to set the colors of the ellipses
                    if color is None:
                        default_palette = plt.rcParams['axes.prop_cycle']
                        default_color_cycle = cycler(
                            colors=default_palette.by_key()['color'])
                        color_cycle = default_color_cycle()
                    else:
                        custom_color_cycle = cycler(colors=color)
                        color_cycle = custom_color_cycle()

                    for i in g.hue_names:
                        cc = next(color_cycle)

                        if ell_fill:
                            ell_facecolor = cc['colors']
                        else:
                            ell_facecolor = "None"

                        _confidence_ellipse(
                            x=x_scores_df_.loc[x_scores_df_[
                                hue] == int(i)][x_lv],
                            y=x_scores_df_.loc[x_scores_df_[
                                hue] == int(i)][y_lv],
                            n_std=n_std,
                            ax=ax,
                            # keep these hardcoded for the time being
                            edgecolor=cc['colors'],
                            linestyle='-',
                            linewidth=ell_linewidth,
                            facecolor=ell_facecolor,
                            alpha=ell_alpha,
                        )

            #########

        return fig, g

    def x_scores_plot(
        self,
        x_lv=1,
        y_lv=2,
        # hue=None,
        figsize=None,
        hue_order=None,
        color=None,
        alpha=1,
        font_scale=1,
        markers=None,
        markersize=35,
        markeredgecolor='white',
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        # show_percent=False,
        legend='full',
        legend_facecolor=None,
        legend_alpha=None,
        legend_title='Class',
        target_labels=None,
        legend_pos='best',
        title=None,
        title_y=1,
        ax=None,
        conf_ellipse=True,
        n_std=2,
        ell_linewidth=1,
        ell_fill=True,
        ell_alpha=0.25,
        **kwargs
    ):
        """Discriminant analysis plot using PLS.
        """
        # pls x_scores dataframe
        x_scores_df_copy_ = self.x_scores_df_.copy()

        if isinstance(x_lv, int):
            x_lv = f'LV{x_lv}'
        if isinstance(y_lv, int):
            y_lv = f'LV{y_lv}'

        # target labels is a dictionary that maps the integer target values
        # (0, 1, ...) to the specified string values
        if target_labels is None:
            target = self.y_train
        else:
            target = list(map(target_labels.get, self.y_train))

        if legend_title is None:
            x_scores_df_copy_['Class'] = target
            hue = 'Class'
        else:
            x_scores_df_copy_[legend_title] = target
            hue = legend_title

        if markers is None:
            style = None
        else:
            style = hue

        fig, ax = plt.subplots(figsize=figsize)

        with sns.plotting_context('notebook', font_scale=font_scale):
            sns.scatterplot(
                data=x_scores_df_copy_,
                x=x_lv,
                y=y_lv,
                hue=hue,
                hue_order=hue_order,
                palette=color,
                alpha=alpha,
                style=style,
                markers=markers,
                s=markersize,
                edgecolor=markeredgecolor,
                legend=legend,
                ax=ax,
                **kwargs
            )

            # set x and y axes labels
            if xlabel is None:
                xlabel = ax.get_xlabel()
            if ylabel is None:
                ylabel = ax.get_ylabel()
            # if show_percent:
            #     x_percent = self.variance_df_.loc['Explained variance ratio',
            #                                       x_lv]
            #     y_percent = self.variance_df_.loc['Explained variance ratio',
            #                                       y_lv]
            #     xlabel = f'{xlabel} ({x_percent:.2%})'
            #     ylabel = f'{ylabel} ({y_percent:.2%})'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if legend:
                ax.legend(
                    facecolor=legend_facecolor,
                    framealpha=legend_alpha,
                    loc=legend_pos,
                    title=legend_title
                )

            # error ellipses
            if conf_ellipse:

                # set color cyclers to set the colors of the ellipses
                if color is None:
                    default_palette = plt.rcParams['axes.prop_cycle']
                    default_color_cycle = cycler(
                        colors=default_palette.by_key()['color'])
                    color_cycle = default_color_cycle()
                else:
                    custom_color_cycle = cycler(colors=color)
                    color_cycle = custom_color_cycle()

                for i in x_scores_df_copy_[hue].unique():
                    cc = next(color_cycle)

                    if ell_fill:
                        ell_facecolor = cc['colors']
                    else:
                        ell_facecolor = "None"

                    _confidence_ellipse(
                        x=x_scores_df_copy_.loc[x_scores_df_copy_[
                            hue] == i][x_lv],
                        y=x_scores_df_copy_.loc[x_scores_df_copy_[
                            hue] == i][y_lv],
                        n_std=n_std,
                        ax=ax,
                        # keep these hardcoded for the time being
                        edgecolor=cc['colors'],
                        linestyle='-',
                        linewidth=ell_linewidth,
                        facecolor=ell_facecolor,
                        alpha=ell_alpha,
                    )

            #####

            # get the figure of the current axes object to return it for
            # convenience after plotting and set title
            # fig = ax.get_figure()
            fig.suptitle(title, y=title_y)

        return fig, ax

    def y_scores_pairplot(
        self,
        n_lvs=None,
        x_vars=None,
        y_vars=None,
        # show_percent=False,
        # hue=None,
        hue_order=None,
        color=None,
        alpha=1,
        markers=None,
        markersize=30,
        markeredgecolor='white',
        height=2.5,
        aspect=1,
        title=None,
        title_y=1,
        font_scale=1,
        legend=True,
        legend_title=None,
        target_labels=None,
        diag_kind=None,
        plot_kws=None,
        diag_kws=None,
        grid_kws=None,
        conf_ellipse=True,
        n_std=2,
        ell_linewidth=1,
        ell_fill=True,
        ell_alpha=0.25,
        **kwargs
    ):
        """Discriminant analysis pairplot using PLS.
        """
        # pls x_scores dataframe
        y_scores_df_ = self._y_scores().iloc[:, :n_lvs]

        # target labels is a dictionary that maps the integer target values
        # (0, 1, ...) to teh specified string values
        if target_labels is None:
            target = self.y_train
        else:
            target = list(map(target_labels.get, self.y_train))

        if legend_title is None:
            y_scores_df_['Target'] = target
            hue = 'Target'
        else:
            y_scores_df_[legend_title] = target
            hue = legend_title

        # add items to plot_kws dictionary
        if plot_kws is None:
            plot_kws = {}
        plot_kws.update({
            's': markersize,
            'alpha': alpha,
            'edgecolor': markeredgecolor,
        })

        # use this to set the scale off all fonts in the plot
        with sns.plotting_context('notebook', font_scale=font_scale):
            g = sns.pairplot(
                y_scores_df_,
                x_vars=x_vars,
                y_vars=y_vars,
                hue=hue,
                hue_order=hue_order,
                palette=color,
                markers=markers,
                height=height,
                aspect=aspect,
                diag_kind=diag_kind,
                plot_kws=plot_kws,
                diag_kws=diag_kws,
                grid_kws=grid_kws,
                **kwargs
            )

            # get the figure of the current PairGrid object to return it for
            # convenience after plotting and set title
            fig = g.fig
            fig.suptitle(title, y=title_y)

            plt.setp(g._legend.get_title(), fontsize=12*font_scale)
            g._legend.set_bbox_to_anchor((1., 0.5))

            if legend is False:
                g._legend.remove()

            # error ellipses

            lvs = list(itertools.product(g.x_vars, g.y_vars))
            lvs_axs = list(el for el in zip(lvs, g.axes.flatten()))
            non_diag_lvs_axs = [el for el in lvs_axs if el[0][0] != el[0][1]]

            if conf_ellipse:

                for (y_lv, x_lv), ax in non_diag_lvs_axs:

                    # set color cyclers to set the colors of the ellipses
                    if color is None:
                        default_palette = plt.rcParams['axes.prop_cycle']
                        default_color_cycle = cycler(
                            colors=default_palette.by_key()['color'])
                        color_cycle = default_color_cycle()
                    else:
                        custom_color_cycle = cycler(colors=color)
                        color_cycle = custom_color_cycle()

                    for i in g.hue_names:
                        cc = next(color_cycle)

                        if ell_fill:
                            ell_facecolor = cc['colors']
                        else:
                            ell_facecolor = "None"

                        _confidence_ellipse(
                            x=y_scores_df_.loc[y_scores_df_[
                                hue] == int(i)][x_lv],
                            y=y_scores_df_.loc[y_scores_df_[
                                hue] == int(i)][y_lv],
                            n_std=n_std,
                            ax=ax,
                            # keep these hardcoded for the time being
                            edgecolor=cc['colors'],
                            linestyle='-',
                            linewidth=ell_linewidth,
                            facecolor=ell_facecolor,
                            alpha=ell_alpha,
                        )

            #########

        return fig, g

    def y_scores_plot(
        self,
        x_lv=1,
        y_lv=2,
        # hue=None,
        hue_order=None,
        color=None,
        alpha=1,
        font_scale=1,
        markers=None,
        markersize=35,
        markeredgecolor='white',
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        # show_percent=False,
        legend='full',
        legend_facecolor=None,
        legend_alpha=None,
        legend_title=None,
        target_labels=None,
        legend_pos='best',
        title=None,
        title_y=1,
        ax=None,
        conf_ellipse=True,
        n_std=2,
        ell_linewidth=1,
        ell_fill=True,
        ell_alpha=0.25,
        **kwargs
    ):
        """Discriminant analysis plot using PLS.
        """
        # pls x_scores dataframe
        y_scores_df_copy_ = self.y_scores_df_.copy()

        if isinstance(y_lv, int):
            x_lv = f'LV{x_lv}'
        if isinstance(y_lv, int):
            y_lv = f'LV{y_lv}'

        # target labels is a dictionary that maps the integer target values
        # (0, 1, ...) to the specified string values
        if target_labels is None:
            target = self.y_train
        else:
            target = list(map(target_labels.get, self.y_train))

        if legend_title is None:
            y_scores_df_copy_['Target'] = target
            hue = 'Target'
        else:
            y_scores_df_copy_[legend_title] = target
            hue = legend_title

        if markers is None:
            style = None
        else:
            style = hue

        with sns.plotting_context('notebook', font_scale=font_scale):
            ax = sns.scatterplot(
                data=y_scores_df_copy_,
                x=x_lv,
                y=y_lv,
                hue=hue,
                hue_order=hue_order,
                palette=color,
                alpha=alpha,
                style=style,
                markers=markers,
                s=markersize,
                edgecolor=markeredgecolor,
                legend=legend,
                ax=ax,
                **kwargs
            )

            # set x and y axes labels
            if xlabel is None:
                xlabel = ax.get_xlabel()
            if ylabel is None:
                ylabel = ax.get_ylabel()
            # if show_percent:
            #     x_percent = self.variance_df_.loc['Explained variance ratio',
            #                                       x_lv]
            #     y_percent = self.variance_df_.loc['Explained variance ratio',
            #                                       y_lv]
            #     xlabel = f'{xlabel} ({x_percent:.2%})'
            #     ylabel = f'{ylabel} ({y_percent:.2%})'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if legend:
                ax.legend(
                    facecolor=legend_facecolor,
                    framealpha=legend_alpha,
                    loc=legend_pos,
                    title=legend_title
                )

            # error ellipses
            if conf_ellipse:

                # set color cyclers to set the colors of the ellipses
                if color is None:
                    default_palette = plt.rcParams['axes.prop_cycle']
                    default_color_cycle = cycler(
                        colors=default_palette.by_key()['color'])
                    color_cycle = default_color_cycle()
                else:
                    custom_color_cycle = cycler(colors=color)
                    color_cycle = custom_color_cycle()

                for i in y_scores_df_copy_[hue].unique():
                    cc = next(color_cycle)

                    if ell_fill:
                        ell_facecolor = cc['colors']
                    else:
                        ell_facecolor = "None"

                    _confidence_ellipse(
                        x=y_scores_df_copy_.loc[y_scores_df_copy_[
                            hue] == i][x_lv],
                        y=y_scores_df_copy_.loc[y_scores_df_copy_[
                            hue] == i][y_lv],
                        n_std=n_std,
                        ax=ax,
                        # keep these hardcoded for the time being
                        edgecolor=cc['colors'],
                        linestyle='-',
                        linewidth=ell_linewidth,
                        facecolor=ell_facecolor,
                        alpha=ell_alpha,
                    )

            #####

            # get the figure of the current axes object to return it for
            # convenience after plotting and set title
            fig = ax.get_figure()
            fig.suptitle(title, y=title_y)

        return fig, ax

    def coef_plot(
        self,
        figsize=None,
        color=None,
        xlabel=None,
        ylabel=None,
        font_scale=1,
        title=None,
        title_y=1,
        legend=False,
        legend_facecolor=None,
        legend_alpha=None,
        legend_pos='best',
        **kwargs
    ):
        """Plot of the PLS coefficients.
        """
        if color is None:
            color = 'tab:blue'

        with sns.plotting_context('notebook', font_scale=font_scale):
            # plot
            ax = self.coef_df_.plot(
                figsize=figsize,
                color=color,
                legend=False,
                **kwargs
            )

            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)

            if legend:
                # set the correct lines and symbols to the legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, facecolor=legend_facecolor,
                          framealpha=legend_alpha,
                          loc=legend_pos)

            fig = ax.get_figure()
            fig.suptitle(title, y=title_y)

        return fig, ax

    # def y_scores_plot(
    #     self,
    # ):
    #     """Plot the y_scores vs the target.
    #     """
    #     y_scores_df_ = pd.DataFrame(
    #         self.pls.y_scores_
    #     )
    #     plt.scatter(
    #         y_scores_df_.iloc[:, 0],
    #         y_scores_df_.iloc[:, 2]
    #     )
    #     return y_scores_df_

    def _rmsecv(
        self,
        # cv_n_components=None
    ):
        """Return a list of the RMSE for each number of components using
        cross-validation.
        """
        # if cv_n_components is None:
        #     cv_n_components = self.cv_n_components

        rmse_list = []

        for i in range(self.cv_n_components):
            # plsi, plsi_fit = self._apply_PLS(
            #     self.x_train,
            #     self.y_train,
            #     n_components=i + 1,
            #     max_iter=self._max_iter,
            #     scale=self._scale,
            #     tol=self._tol
            # )

            y_cv = model_selection.cross_val_predict(
                cross_decomposition.PLSRegression(n_components=i + 1),
                self.x_train,
                self.y_train,
                # groups=self.y_train,
                cv=self.cv
            )

            # print(y_cv)

            rmse = mean_squared_error(
                self.y_train,
                y_cv,
                squared=True)

            # y_pred = plsi.predict(self.x_test)
            # rmse = mean_squared_error(
            #     y_pred,
            #     self.y_test,
            #     squared=True)

            rmse_list.append(rmse)

        return rmse_list

    def rmsecv_df(self,
                  # cv_n_components=None
                  ):
        """Return a dataframe of the RMSE for each number of components using
        cross-validation.
        """
        # if cv_n_components is None:
        #     cv_n_components = self.cv_n_components

        rmse_list = self._rmsecv()

        rmse_df = pd.DataFrame(
            {'RMSE': rmse_list},
            index=[i + 1 for i in range(self.cv_n_components)]
        )
        rmse_df.index.name = 'Components'

        return rmse_df

    def rmsecv_plot(
        self,
        # cv_n_components=None,
        figsize=None,
        color=None,
        marker='o',
        markersize=None,
        label=None,
        title=None,
        title_y=1,
        xlabel='Number of components',
        ylabel='RMSE',
        xlim=None,
        ylim=None,
        font_scale=1,
        legend=False,
        legend_facecolor=None,
        legend_alpha=None,
        legend_pos='upper right',
        **kwargs
    ):
        """Plot the RMSE vs number of components for cross-validation and
        indicate the minimum.
        """
        # if cv_n_components is None:
        #     cv_n_components = self.cv_n_components

        rmsecv_df = self.rmsecv_df()

        if color is None:
            color = 'tab:blue'

        with sns.plotting_context('notebook', font_scale=font_scale):
            # plot
            ax = rmsecv_df.plot(
                figsize=figsize,
                color=color,
                markersize=markersize,
                legend=False,
                **kwargs
            )

            # set x and y labels
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # set xticks as integers
            ax.set_xticks(range(1, self.cv_n_components + 1))

            # set x and y lims
            # (if set in df.plot, they are overriden by set_xticks)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # set the desired markers and labels for each line
            for i, line in enumerate(ax.get_lines()):
                if label is not None:
                    line.set_label(label[i])
                if marker is not None:
                    line.set_marker(marker[i])

            if legend:
                # set the correct lines and symbols to the legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, facecolor=legend_facecolor,
                          framealpha=legend_alpha,
                          loc=legend_pos)

            # get the figure of the current axes object to return it for
            # convenience after plotting and set title
            fig = ax.get_figure()
            fig.suptitle(title, y=title_y)

        return fig, ax

    def rmsecv_min(self):
        """Get the number of components for which the cross-validation RMSE is
        minimum.
        """
        rmsecv_df = self.rmsecv_df()

        return rmsecv_df['RMSE'].idxmin()

    def confusion_matrix(
        self,
        margins=False,
        target_names=None
    ):
        """Calculate the confusion matrix for the test data and return it as
        a pandas dataframe. Also display a report contining the main
        classification metrics using sklearn.metrics.classification_report.
        """
        data = {'Actual': self.y_test,
                'Predicted': self.prediction_df_['Rounded_prediction']}

        df = pd.DataFrame(data, columns=['Actual', 'Predicted'])

        conf_mat = pd.crosstab(
            df['Actual'],
            df['Predicted'],
            rownames=['Actual'],
            colnames=['Predicted'],
            margins=margins
        )

        # change column and index names
        if target_names is not None:
            cols = conf_mat.columns
            idx = conf_mat.index
            conf_mat.rename(
                columns=dict(zip(cols, target_names)),
                index=dict(zip(idx, target_names)),
                inplace=True
            )

        report = classification_report(
            df['Actual'],
            df['Predicted'],
            target_names=target_names
        )

        print('Report:')
        print(report)

        return conf_mat

    def confusion_matrix_fancy(
        self,
        margins=False,
        target_names=None,
        figsize=None,
        cmap=None,
        cbar=True,
        font_scale=1,
        xlabel='auto',
        ylabel='auto',
        xticklabels_rotation='horizontal',
        yticklabels_rotation='vertical',
        xticklabels_ha='center',
        yticklabels_va='center',
        title=None,
        title_y=1,
        **kwargs
    ):
        """A beautified version of the confusion matrix using seaborn. Also
        display a report contining the main classification metrics using
        sklearn.metrics.classification_report.
        """
        conf_mat = self.confusion_matrix(
            margins=margins,
            target_names=target_names
        )

        if target_names is not None:
            xticklabels = target_names
            yticklabels = target_names
        else:
            xticklabels = 'auto'
            yticklabels = 'auto'

        with sns.plotting_context('notebook', font_scale=font_scale):
            fig, ax = plt.subplots(figsize=figsize)

            sns.heatmap(
                data=conf_mat,
                ax=ax,
                cmap=cmap,
                cbar=cbar,
                annot=True,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                square=True,
                **kwargs
            )

            # set x and y labels
            if xlabel != 'auto':
                ax.set_xlabel(xlabel)
            if ylabel != 'auto':
                ax.set_ylabel(ylabel)

            # customize tick labels
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=xticklabels_rotation,
                ha=xticklabels_ha
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=yticklabels_rotation,
                va=yticklabels_va
            )

            fig.suptitle(title, y=title_y)

        return fig, ax

    def roc(
        self,
        figsize=None,
        color=None,
        xlabel='False positive rate',
        ylabel='True positive rate',
        font_scale=1,
        legend=True,
        legend_facecolor=None,
        legend_alpha=None,
        legend_pos='lower right',
        title=None,
        title_y=1,
        **kwargs
    ):
        """Plot the Receiver Operating Characteristic (ROC) curve.
        """
        # ################## SOS #####################
        # not sure if prediction or rounded_prediction should be used!!!!!
        y_pred = self.prediction_df_['Rounded_prediction']
        # y_pred = self.prediction_df_['Prediction']
        fpr, tpr, thr = roc_curve(self.y_test, y_pred, drop_intermediate=False)
        # print(fpr, tpr, thr)

        auc = roc_auc_score(self.y_test, y_pred)

        with sns.plotting_context('notebook', font_scale=font_scale):
            fig, ax = plt.subplots(figsize=figsize)

            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.plot(
                fpr,
                tpr,
                label=f'ROC curve (AUC = {auc:.2f})',
                color=color,
                **kwargs
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if legend:
                # set the correct lines and symbols to the legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, facecolor=legend_facecolor,
                          framealpha=legend_alpha,
                          loc=legend_pos)

            fig.suptitle(title, y=title_y)

        return fig, ax


##########################################
def _confidence_ellipse(
    x,
    y,
    ax,
    n_std=3.0,
    facecolor='none',
    **kwargs
):
    """Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)

    num = cov[0, 1]
    den = np.sqrt(cov[0, 0] * cov[1, 1])

    # avoid division with zero
    if den == 0:
        den = 0.000000000000001

    pearson = num / den

    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

#############################################
