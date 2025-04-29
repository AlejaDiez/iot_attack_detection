import { Injectable, Signal } from "@angular/core";
import { toSignal } from "@angular/core/rxjs-interop";
import { NavigationEnd, Router as NavigationRouter } from "@angular/router";
import { Device } from "@models/device";
import { Network } from "@models/network";
import { Node, NodeType } from "@models/node";
import { Position } from "@models/position";
import { Router } from "@models/router";
import {
    AddNodeDialogComponent,
    AddNodeDialogContext,
} from "@routes/dialogs/add-node-dialog.component";
import {
    DeleteNodeDialogComponent,
    DeleteNodeDialogContext,
} from "@routes/dialogs/delete-node-dialog.component";
import { ConfigService } from "@services/config.service";
import { StateService } from "@services/state.service";
import { HlmDialogService } from "@spartan-ng/ui-dialog-helm";
import { dump, load } from "js-yaml";
import { toast } from "ngx-sonner";
import { filter, map } from "rxjs";

/**
 * Clase que permite gestionar la red de dispositivos.
 */
@Injectable({ providedIn: "root" })
export class NetworkService {
    /** Red de dispositivos */
    private readonly _network: Network = new Network();
    /** Obtiene los nodos de la red de dispositivos */
    public get nodes(): Node[] {
        return this._network.nodes;
    }
    /** Obtiene el router de la red de dispositivos */
    public get router(): Router | undefined {
        return this._network.router;
    }
    /** Obtiene los dispositivos de la red de dispositivos */
    public get devices(): Device[] | undefined {
        return this._network.devices;
    }
    /** Dirección MAC del nodo seleccionado */
    public readonly focusedNode: Signal<Node | undefined>;

    /**
     * Constructor del servicio de gestión de la red de dispositivos.
     *
     * @param navigationRouter Servicio de navegación de la aplicación
     * @param config Servicio de configuración de la aplicación
     * @param dialog Servicio de diálogos de la aplicación
     */
    public constructor(
        public readonly navigationRouter: NavigationRouter,
        public readonly dialog: HlmDialogService,
        public readonly config: ConfigService,
        public readonly state: StateService,
    ) {
        // Obtener el nodo seleccionado
        this.focusedNode = toSignal(
            this.navigationRouter.events.pipe(
                filter((event) => event instanceof NavigationEnd),
                map(({ url }) =>
                    url.split("/").length > 1 ? url.split("/")[1] : undefined,
                ),
                map((mac) => this._network.getNode(mac ?? "")),
            ),
        );
    }

    /**
     * Crea una nueva red de dispositivos.
     */
    public new(): void {
        this._network.reset();
        // Comprobar si hay un nodo seleccionado
        if (this.focusedNode()) this.navigationRouter.navigate([""]);
        // Guardar el estado de la red de dispositivos
        this.state.reset(false);
        toast.success("Proyecto creado correctamente.");
    }

    /**
     * Carga una red de dispositivos desde un archivo.
     */
    public loadFromFile(): void {
        toast.promise(ConfigService.openFile(".yaml"), {
            loading: "Importando proyecto...",
            success: (data: string) => {
                this._network.fromObject(load(data));
                // Comprobar si hay un nodo seleccionado
                if (
                    this.focusedNode() &&
                    !this._network.getNode(this.focusedNode()!.mac ?? "")
                )
                    this.navigationRouter.navigate([""]);
                // Guardar el estado de la red de dispositivos
                this.state.reset(false);
                this.state.setState(this._network.toObject(), false);
                return "Proyecto importado correctamente.";
            },
            error: () => "No se ha podido importar el proyecto.",
        });
    }

    /**
     * Guarda la red de dispositivos en un archivo.
     */
    public saveToFile(): void {
        toast.promise(
            new Promise<[string, string, string]>((resolve, reject) => {
                const name: string = [
                    "iot",
                    "simulator",
                    ...new Date().toISOString().split(/T|\./g, 2),
                ]
                    .join("_")
                    .replace(/-|:/g, "")
                    .concat(".yaml");
                const content: string = dump(this._network.toObject(), {
                    noCompatMode: true,
                    forceQuotes: true,
                });
                const type: string = "application/x-yaml";

                resolve([name, content, type]);
            }).then(([name, content, type]) =>
                ConfigService.saveFile(name, content, type),
            ),
            {
                loading: "Exportando proyecto...",
                success: () => {
                    // Guardar el estado de la red de dispositivos
                    this.state.replaceState(this._network.toObject(), false);
                    return "Proyecto exportado correctamente.";
                },
                error: () => "No se ha podido exportar el proyecto.",
            },
        );
    }

    /**
     * Añade un nodo a la red de dispositivos.
     *
     * @param type Tipo de nodo
     * @param position Posición del nodo
     */
    public addNode(
        type?: NodeType | readonly NodeType[] | NodeType[],
        position?: Position,
    ): Promise<boolean> {
        return new Promise((resolve, reject) => {
            if (
                this._network.router &&
                (Array.isArray(type)
                    ? NodeType.RouterTypes.some((t) => type.includes(t))
                    : NodeType.RouterTypes.includes(type as NodeType))
            ) {
                toast.error("No se puede agregar más de un router a la red.");
                return reject();
            }

            this.dialog
                .open(AddNodeDialogComponent, { context: { type } })
                .closed$.subscribe((context: AddNodeDialogContext) => {
                    if (!context) return resolve(false);

                    this._network.addNode(
                        NodeType.RouterTypes.includes(context.type)
                            ? new Router(context.name, position)
                            : new Device(
                                  context.name,
                                  context.type as any,
                                  position,
                              ),
                    );
                    // Guardar el estado de la red de dispositivos
                    this.state.setState(this._network.toObject(), false);
                    toast.success(
                        `Se ha añadido ${context.name} correctamente.`,
                    );
                    return resolve(true);
                });
        });
    }

    /**
     * Elimina un nodo de la red de dispositivos.
     *
     * @param mac Dirección MAC del nodo
     */
    public deleteNode(mac: string): Promise<boolean> {
        return new Promise((resolve, reject) => {
            const node = this._network.getNode(mac);

            if (!node) {
                toast.error(
                    "No existe un nodo con la dirección MAC especificada.",
                );
                return reject();
            }

            this.dialog
                .open(DeleteNodeDialogComponent, {
                    context: { node },
                })
                .closed$.subscribe((context: DeleteNodeDialogContext) => {
                    if (!context) return resolve(false);

                    this._network.deleteNode(context.node.mac);
                    // Comprobar si el nodo eliminado es el nodo seleccionado
                    if (this.focusedNode()?.mac === node.mac)
                        this.navigationRouter.navigate([""]);
                    // Guardar el estado de la red de dispositivos
                    this.state.setState(this._network.toObject(), false);
                    toast.success(
                        `Se ha eliminado ${context.node.name} correctamente.`,
                    );
                    return resolve(true);
                });
        });
    }

    /**
     * Obtiene un nodo de la red de dispositivos.
     *
     * @param mac Dirección MAC del nodo
     * @returns Nodo obtenido
     */
    public getNode(mac: string): Node | undefined {
        return this._network.getNode(mac);
    }

    /**
     * Obtiene los nodos conectados a un nodo de la red de dispositivos.
     *
     * @param mac Dirección MAC del nodo
     * @returns Nodos conectados
     */
    public getConnectedNodes(mac?: string): Node[] {
        return this._network.getConnectedNodes(mac);
    }
}
