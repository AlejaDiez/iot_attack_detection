import { Injectable } from "@angular/core";
import {
    AddNodeDialogComponent,
    AddNodeDialogContext,
} from "@components/dialog/add-node-dialog.component";
import {
    DeleteNodeDialogComponent,
    DeleteNodeDialogContext,
} from "@components/dialog/delete-node-dialog.component";
import { Connection } from "@models/connection";
import { Device } from "@models/device";
import { Node, NodeType } from "@models/node";
import { Position } from "@models/position";
import { Router } from "@models/router";
import { HlmDialogService } from "@spartan-ng/ui-dialog-helm";
import { dump, load } from "js-yaml";
import { toast } from "ngx-sonner";

@Injectable({ providedIn: "root" })
export class NetworkManagerService {
    private _nodes: Map<string, Node> = new Map();
    private _routerMac?: string;
    public get nodes(): Node[] {
        return [...this._nodes.values()];
    }
    public get router(): Router | undefined {
        return this._routerMac
            ? (this._nodes.get(this._routerMac) as Router)
            : undefined;
    }
    public get devices(): Device[] | undefined {
        const devices = this.nodes.filter(
            (node) => node.type !== NodeType.ROUTER,
        ) as Device[];

        return devices.length > 0 ? devices : undefined;
    }
    public get connections(): Connection[] | undefined {
        return this.router?.connections;
    }

    constructor(private _dialog: HlmDialogService) {}

    private _addNode(name: string, type: NodeType, position?: Position): Node {
        if (this._routerMac && type === NodeType.ROUTER)
            throw new Error("Cannot add more than one router to the network");

        let node;

        if (type === NodeType.ROUTER) {
            node = new Router(name, position);
            this._routerMac = node.mac;
        } else {
            node = new Device(name, type, position);
        }
        this._nodes.set(node.mac, node);
        return node;
    }

    private _deleteNode(mac: string): boolean {
        if (!this.exists(mac))
            throw new Error(
                "Does not exist a node with the specified MAC address",
            );

        if (mac === this._routerMac) this._routerMac = undefined;
        return this._nodes.delete(mac);
    }

    public newNetwork(): void {
        this._nodes.clear();
        this._routerMac = undefined;
        console.log("New network created");
        toast.success("Se ha creado una nueva red correctamente.");
    }

    public exportNetwork(): void {
        toast.promise(
            new Promise<void>(async (resolve) => {
                const date: Date = new Date();
                const data: object = {
                    exportedAt: date.toISOString(),
                    router: this.router?.toObject(),
                    devices: this.devices?.map((e) => e.toObject()),
                };
                const fileContent: string = dump(data, {
                    noCompatMode: true,
                    forceQuotes: true,
                });
                const fileName: string = `iot_simulator_${date.getDate().toString().padStart(2, "0")}${(date.getMonth() + 1).toString().padStart(2, "0")}${date.getFullYear()}${date.getHours().toString().padStart(2, "0")}${date.getMinutes().toString().padStart(2, "0")}${date.getSeconds().toString().padStart(2, "0")}.yaml`;
                const blob: Blob = new Blob([fileContent], {
                    type: "application/x-yaml",
                });

                // Set a short delay to show the loading toast
                await new Promise((_) => setTimeout(_, 1000));

                // Create a invisible link to download the file
                const link: HTMLAnchorElement = document.createElement("a");

                link.style.display = "none";
                link.href = URL.createObjectURL(blob);
                link.download = fileName;
                link.click();
                URL.revokeObjectURL(link.href);
                link.remove();

                // Finish the promise
                resolve();
            }),
            {
                loading: "Exportando configuración de la red...",
                success: () => {
                    console.log("Network configuration exported successfully.");
                    return "Configuración de la red exportada correctamente.";
                },
                error: () => {
                    console.error("Failed to export network configuration.");
                    return "No se ha podido exportar la configuración de la red.";
                },
            },
        );
    }

    public importNetwork(): void {
        const readFile = (file: File) => {
            toast.promise(
                new Promise<void>(async (resolve, reject) => {
                    const reader = new FileReader();

                    reader.onload = async () => {
                        const data: any = load(reader.result as string);

                        // Delete the current network
                        this._nodes.clear();
                        this._routerMac = undefined;

                        // Set a short delay to show the loading toast
                        await new Promise((_) => setTimeout(_, 1000));

                        try {
                            // Add the router
                            if (data.router) {
                                const router = Router.fromObject(data.router);

                                this._nodes.set(router.mac, router);
                                this._routerMac = router.mac;
                            }

                            // Add the devices
                            if (data.devices) {
                                data.devices.forEach((obj: any) => {
                                    const device = Device.fromObject(obj);

                                    if (obj.connection)
                                        device.connect(
                                            this.router!,
                                            obj.connection,
                                        );
                                    this._nodes.set(device.mac, device);
                                });
                            }
                        } catch (e) {
                            this._nodes.clear();
                            this._routerMac = undefined;
                            reject();
                        }
                        // Finish the promise
                        resolve();
                    };
                    // Read the file as text
                    reader.readAsText(file);
                }),
                {
                    loading: "Importando archivo configuración de la red...",
                    success: () => {
                        console.log(
                            "Network configuration imported successfully.",
                        );
                        return "Configuración de la red importada correctamente.";
                    },
                    error: () => {
                        console.error(
                            "Failed to import network configuration.",
                        );
                        return "No se ha podido importar la configuración de la red.";
                    },
                },
            );
        };

        // Create a invisible input to select the file
        const input: HTMLInputElement = document.createElement("input");

        input.style.display = "none";
        input.type = "file";
        input.accept = ".yaml";
        input.click();
        input.onchange = (event: any) => {
            const file: File = event.target.files[0];

            if (file) readFile(file);
            input.value = "";
            input.remove();
        };
    }

    public addNode(type?: NodeType, position?: Position) {
        if (this._routerMac && type === NodeType.ROUTER) {
            console.error("Cannot add more than one router to the network");
            toast.error("No se puede agregar más de un router a la red.");
            return;
        }

        this._dialog
            .open(AddNodeDialogComponent, { context: { type } })
            .closed$.subscribe((context: AddNodeDialogContext) => {
                if (!context) return;

                const node = this._addNode(
                    context.name,
                    context.type,
                    position,
                );

                console.log("Added new node to the network", node);
                toast.success(`Se ha añadido ${context.name} correctamente.`);
            });
    }

    public deleteNode(mac: string) {
        if (!this.exists(mac)) {
            console.error(
                "Does not exist a node with the specified MAC address",
            );
            toast.error("No existe un nodo con la dirección MAC especificada.");
            return;
        }

        this._dialog
            .open(DeleteNodeDialogComponent, {
                context: { node: this.findByMac(mac) },
            })
            .closed$.subscribe((context: DeleteNodeDialogContext) => {
                if (!context) return;

                this._deleteNode(context.node.mac);
                console.log("Node deleted from the network", context.node);
                toast.success(
                    `Se ha eliminado ${context.node.name} correctamente.`,
                );
            });
    }

    public exists(mac: string): boolean {
        return this._nodes.has(mac);
    }

    public findByMac(mac: string): Node {
        if (!this.exists(mac))
            throw new Error(
                "Does not exist a node with the specified MAC address",
            );
        return this._nodes.get(mac)!;
    }
}
