import { Injectable } from "@angular/core";
import { NavigationEnd, Router as NavigationRouter } from "@angular/router";
import { Device } from "@models/device";
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
import { HlmDialogService } from "@spartan-ng/ui-dialog-helm";
import { dump, load } from "js-yaml";
import { toast } from "ngx-sonner";
import { debounceTime, filter, map, skip } from "rxjs";
@Injectable({ providedIn: "root" })
export class NetworkManagerService {
    /** Nodos de la red */
    private readonly _nodes: Map<string, Node> = new Map<string, Node>();
    /** Dirección MAC del router de la red */
    private _routerMac?: string;
    /** Obtiene los nodos de la red */
    public get nodes(): Node[] {
        return [...this._nodes.values()];
    }
    /** Obtiene el router de la red */
    public get router(): Router | undefined {
        return this._routerMac
            ? (this._nodes.get(this._routerMac) as Router)
            : undefined;
    }
    /** Obtiene los dispositivos de la red */
    public get devices(): Device[] | undefined {
        const devices = this.nodes.filter(
            (node) => node.type !== NodeType.ROUTER,
        ) as Device[];

        return devices.length > 0 ? devices : undefined;
    }
    /** Dirección MAC del nodo seleccionado */
    private _focusedNode: string | undefined;
    /** Dirección MAC del nodo seleccionado */
    public get focusedNode(): string | undefined {
        return this._focusedNode;
    }

    /**
     * Constructor del servicio de gestión de la red de dispositivos.
     *
     * @param navigationRouter Servicio de navegación de la aplicación
     * @param config Servicio de configuración de la aplicación
     * @param dialog Servicio de diálogos de la aplicación
     */
    public constructor(
        public readonly navigationRouter: NavigationRouter,
        public readonly config: ConfigService,
        public readonly dialog: HlmDialogService,
    ) {
        this.navigationRouter.events
            .pipe(
                filter((event) => event instanceof NavigationEnd),
                map(({ url }) =>
                    url.split("/").length > 1 ? url.split("/")[1] : undefined,
                ),
            )
            .subscribe((url) => (this._focusedNode = url));
        this.config.stateManager.state$.subscribe((state) =>
            this.fromObject(state),
        );
        this.config.libraryManager.library$
            .pipe(skip(1))
            .subscribe((library) =>
                this.nodes.forEach((e) => e.loadLibrary(library)),
            );
    }

    /**
     * Restablece la red de dispositivos.
     */
    private _reset(state: boolean = true) {
        this._nodes.clear();
        this._routerMac = undefined;
        if (state) this.config.stateManager.reset(false);
    }

    /**
     * Añade un nodo a la red de dispositivos.
     *
     * @param node Nodo a añadir
     */
    private _addNode(node: Node, state: boolean = true): Node {
        if (this._routerMac && node.type === NodeType.ROUTER)
            throw new Error("Cannot add more than one router to the network");

        if (node.type === NodeType.ROUTER) this._routerMac = node.mac;
        this._nodes.set(node.mac, node);
        node.state$
            .pipe(debounceTime(500))
            .subscribe(() =>
                this.config.stateManager.setState(this.toObject(), false),
            );
        node.loadLibrary(this.config.libraryManager.library);
        if (state) this.config.stateManager.setState(this.toObject(), false);
        return node;
    }

    /**
     * Elimina un nodo de la red de dispositivos.
     *
     * @param mac Dirección MAC del nodo
     * @returns Indica si el nodo ha sido eliminado
     */
    private _deleteNode(mac: string, state: boolean = true): boolean {
        const node = this._nodes.get(mac);

        if (!node)
            throw new Error(
                "Does not exist a node with the specified MAC address",
            );
        if (node.mac === this._routerMac) this._routerMac = undefined;
        if (node.type === NodeType.ROUTER)
            (node as Router).removeAllConnections();
        else if (node.connected && this.router)
            (node as Device).disconnect(this.router);
        this._nodes.delete(node.mac);
        if (state) this.config.stateManager.setState(this.toObject(), false);
        return true;
    }

    /**
     * Crea una nueva red de dispositivos.
     */
    public new(): void {
        this._reset();
        // Check if device is selected and exists
        if (this._focusedNode && !this.exists(this._focusedNode))
            this.navigationRouter.navigate([""]);
        toast.success("Proyecto creado correctamente.");
    }

    /**
     * Carga una red de dispositivos desde un archivo.
     */
    public loadFromFile(): void {
        toast.promise(this.config.openFile(".yaml"), {
            loading: "Importando proyecto...",
            success: (data: string) => {
                this.fromObject(load(data));
                this.config.stateManager.reset(false);
                this.config.stateManager.replaceState(this.toObject(), false);
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
                const content: string = dump(this.toObject(), {
                    noCompatMode: true,
                    forceQuotes: true,
                });
                const type: string = "application/x-yaml";

                resolve([name, content, type]);
            }).then(([name, content, type]) =>
                this.config.saveFile(name, content, type),
            ),
            {
                loading: "Exportando proyecto...",
                success: () => {
                    this.config.stateManager.replaceState(
                        this.toObject(),
                        false,
                    );
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
                this._routerMac &&
                (Array.isArray(type)
                    ? type.includes(NodeType.ROUTER)
                    : type === NodeType.ROUTER)
            ) {
                toast.error("No se puede agregar más de un router a la red.");
                return reject();
            }

            this.dialog
                .open(AddNodeDialogComponent, { context: { type } })
                .closed$.subscribe((context: AddNodeDialogContext) => {
                    if (!context) return resolve(false);

                    this._addNode(
                        NodeType.RouterTypes.includes(context.type)
                            ? new Router(context.name, position)
                            : new Device(
                                  context.name,
                                  context.type as any,
                                  position,
                              ),
                    );
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
            if (!this.exists(mac)) {
                toast.error(
                    "No existe un nodo con la dirección MAC especificada.",
                );
                return reject();
            }

            this.dialog
                .open(DeleteNodeDialogComponent, {
                    context: { node: this.findByMac(mac) },
                })
                .closed$.subscribe((context: DeleteNodeDialogContext) => {
                    if (!context) return resolve(false);

                    this._deleteNode(context.node.mac);
                    // Check if device is selected and exists
                    if (this._focusedNode && !this.exists(this._focusedNode))
                        this.navigationRouter.navigate([""]);
                    toast.success(
                        `Se ha eliminado ${context.node.name} correctamente.`,
                    );
                    return resolve(true);
                });
        });
    }

    /**
     * Verifica si existe un nodo en la red de dispositivos.
     *
     * @param mac Dirección MAC del nodo
     * @returns Indica si el nodo existe
     */
    public exists(mac: string): boolean {
        return this._nodes.has(mac);
    }

    /**
     * Obtiene un nodo de la red de dispositivos.
     *
     * @param mac Dirección MAC del nodo
     * @returns Nodo obtenido
     */
    public findByMac(mac: string): Node {
        if (!this.exists(mac))
            throw new Error(
                "Does not exist a node with the specified MAC address",
            );
        return this._nodes.get(mac)!;
    }

    /**
     * Obtiene los nodos conectados a un nodo de la red de dispositivos.
     *
     * @param mac Dirección MAC del nodo para obtener los nodos conectados a este.
     * @returns Nodos conectados
     */
    public getConnectedNodes(mac?: string): Node[] {
        return this.nodes.filter((node) => node.ip && mac !== node.mac);
    }

    /**
     * Transforma la red de dispositivos en un objeto.
     *
     * @returns Objeto de la red de dispositivos
     */
    public toObject(): any {
        return {
            exportedAt: new Date().toISOString(),
            router: this.router?.toObject(),
            devices: this.devices?.map((e) => e.toObject()),
        };
    }

    /**
     * Carga una red de dispositivos desde un objeto.
     *
     * @param object Objeto de la red de dispositivos
     */
    public fromObject(object: any): void {
        this._reset(false);
        // Add the router
        if (object.router) {
            this._addNode(Router.fromObject(object.router), false);
        }
        // Add the devices
        if (object.devices) {
            object.devices.forEach((obj: any) => {
                const device = this._addNode(
                    Device.fromObject(obj),
                    false,
                ) as Device;

                if (obj.connection)
                    device.connect(this.router!, obj.connection);
            });
        }
        // Check if device is selected and exists
        if (this._focusedNode && !this.exists(this._focusedNode))
            this.navigationRouter.navigate([""]);
    }
}
