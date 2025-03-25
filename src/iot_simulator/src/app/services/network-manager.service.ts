import { Injectable } from "@angular/core";
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
import { HlmDialogService } from "@spartan-ng/ui-dialog-helm";
import { toast } from "ngx-sonner";
import { ConfigService } from "./config.service";

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

    /**
     * Constructor del servicio de gestión de la red de dispositivos.
     *
     * @param _dialog Servicio de diálogos de la aplicación
     */
    public constructor(
        private readonly _dialog: HlmDialogService,
        private readonly _config: ConfigService,
    ) {}

    /**
     * Añade un nodo a la red de dispositivos.
     *
     * @param name Nombre del nodo
     * @param type Tipo de nodo
     * @param position Posición del nodo
     * @returns Nodo añadido
     */
    private _addNode(name: string, type: NodeType, position?: Position): Node {
        let node;

        if (this._routerMac && type === NodeType.ROUTER)
            throw new Error("Cannot add more than one router to the network");
        if (type === NodeType.ROUTER) {
            node = new Router(name, position);
            this._routerMac = node.mac;
        } else {
            node = new Device(name, type, position);
        }
        this._nodes.set(node.mac, node);
        return node;
    }

    /**
     * Elimina un nodo de la red de dispositivos.
     *
     * @param mac Dirección MAC del nodo
     * @returns Indica si el nodo ha sido eliminado
     */
    private _deleteNode(mac: string): boolean {
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
        return true;
    }

    /**
     * Crea una nueva red de dispositivos.
     */
    public new() {
        this._nodes.clear();
        this._routerMac = undefined;
        toast.success("Proyecto creado correctamente.");
    }

    /**
     * Carga una red de dispositivos desde un archivo.
     */
    public loadFromFile() {
        toast.promise(this._config.openFile(), {
            loading: "Importando proyecto...",
            success: (data: any) => {
                this.fromObject(data);
                return "Proyecto importado correctamente.";
            },
            error: () => {
                console.error(
                    "Failed to import project, the file is not valid",
                );
                return "No se ha podido importar el proyecto.";
            },
        });
    }

    /**
     * Guarda la red de dispositivos en un archivo.
     */
    public saveToFile() {
        toast.promise(this._config.saveFile(this.toObject()), {
            loading: "Exportando proyecto...",
            success: () => {
                return "Proyecto exportado correctamente.";
            },
            error: () => {
                console.error("Failed to export project");
                return "No se ha podido exportar el proyecto.";
            },
        });
    }

    /**
     * Añade un nodo a la red de dispositivos.
     *
     * @param type Tipo de nodo
     * @param position Posición del nodo
     */
    public addNode(type?: NodeType, position?: Position): Promise<boolean> {
        return new Promise((resolve, reject) => {
            if (this._routerMac && type === NodeType.ROUTER) {
                console.error("Cannot add more than one router to the network");
                toast.error("No se puede agregar más de un router a la red.");
                return reject();
            }
            this._dialog
                .open(AddNodeDialogComponent, { context: { type } })
                .closed$.subscribe((context: AddNodeDialogContext) => {
                    if (!context) return resolve(false);

                    const node = this._addNode(
                        context.name,
                        context.type,
                        position,
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
                console.error(
                    "Does not exist a node with the specified MAC address to delete",
                );
                toast.error(
                    "No existe un nodo con la dirección MAC especificada.",
                );
                return reject();
            }
            this._dialog
                .open(DeleteNodeDialogComponent, {
                    context: { node: this.findByMac(mac) },
                })
                .closed$.subscribe((context: DeleteNodeDialogContext) => {
                    if (!context) return resolve(false);

                    this._deleteNode(context.node.mac);
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
     * @param mac Dirección MAC del nodo
     * @returns Nodos conectados
     */
    public getConnectedNodes(): Node[] {
        return this.nodes.filter((node) => node.ip);
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
        // Reset the network
        this._nodes.clear();
        this._routerMac = undefined;
        // Add the router
        if (object.router) {
            const router = Router.fromObject(object.router);

            this._nodes.set(router.mac, router);
            this._routerMac = router.mac;
        }
        // Add the devices
        if (object.devices) {
            object.devices.forEach((obj: any) => {
                const device = Device.fromObject(obj);

                if (obj.connection)
                    device.connect(this.router!, obj.connection);
                this._nodes.set(device.mac, device);
            });
        }
    }
}
