"""Static diagnostics; optional self-contained interactive HTML surfaces."""
import numpy as np


def plot_results(surfaces, table, predictions, output, *, interactive=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    for name,s in surfaces.items():
        w,d=np.meshgrid(s.moisture,s.density,indexing="ij")
        fig,ax=plt.subplots(figsize=(7,5),layout="constrained")
        p=ax.pcolormesh(w,d,np.ma.masked_invalid(s.z),shading="auto",cmap="viridis")
        fig.colorbar(p,ax=ax,label="Thermal conductivity [W/(m K)]")
        ax.set(xlabel="Gravimetric water content [%]",ylabel="Wet bulk density [g/cm³]",title=name)
        fig.savefig(output/f"{name}_2d.png",dpi=160);plt.close(fig)
        fig=plt.figure(figsize=(8,6),layout="constrained")
        ax=fig.add_subplot(projection="3d")
        ax.plot_surface(w,d,s.z,cmap="viridis",linewidth=0)
        ax.set(xlabel="Water content [%]",ylabel="Wet bulk density [g/cm³]",
               zlabel="Lambda [W/(m K)]",title=name)
        fig.savefig(output/f"{name}_3d.png",dpi=160);plt.close(fig)
        if predictions is not None:
            g=predictions[predictions.model_id==name]
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
            axes[0].scatter(g.observed_w_mk,g.predicted_w_mk,s=15)
            lo=min(g.observed_w_mk.min(),g.predicted_w_mk.min())
            hi=max(g.observed_w_mk.max(),g.predicted_w_mk.max())
            axes[0].plot([lo,hi],[lo,hi],color="black",lw=1)
            axes[0].set(xlabel="Observed [W/(m K)]",ylabel="Predicted [W/(m K)]",title=name)
            axes[1].scatter(g.predicted_w_mk,g.residual_w_mk,s=15)
            axes[1].axhline(0,color="black",lw=1)
            axes[1].set(xlabel="Predicted [W/(m K)]",ylabel="Predicted - observed [W/(m K)]")
            fig.savefig(output/f"{name}_holdout.png",dpi=160);plt.close(fig)
        if interactive:
            try:
                import plotly.graph_objects as go
            except ImportError as exc:
                raise ValueError("Interactive surfaces require the [interactive] extra") from exc
            figure=go.Figure(go.Surface(x=w.T,y=d.T,z=s.z.T))
            figure.update_layout(title=name,scene={"xaxis_title":"Water content [%]",
                "yaxis_title":"Wet bulk density [g/cm3]","zaxis_title":"Lambda [W/(m K)]"})
            figure.write_html(output/f"{name}_3d.html",include_plotlyjs=True,full_html=True)
    if "rmse" in table:
        fig,ax=plt.subplots(figsize=(7,5),layout="constrained")
        ax.scatter(table.rmse,table.pci,c=table.pareto.map({True:"#087e8b",False:"#b4b4b4"}),s=65)
        for row in table.itertuples():
            ax.annotate(row.model_id,(row.rmse,row.pci),xytext=(5,5),textcoords="offset points")
        ax.set(xlabel="Holdout RMSE [W/(m K)]",ylabel="Relative PCI",title="Accuracy and physical consistency")
        fig.savefig(output/"rmse_pci.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout="constrained")
    for ax,col in zip(axes,["mvr","rough_norm","pbv"]):
        ax.bar(table.model_id,table[col],color="#087e8b");ax.set_title(col)
        ax.tick_params(axis="x",rotation=45)
    fig.savefig(output/"pci_components.png",dpi=160);plt.close(fig)
