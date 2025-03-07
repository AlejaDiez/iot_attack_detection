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
import { toast } from "ngx-sonner";

@Injectable({ providedIn: "root" })
export class NetworkManagerService {
    private _nodes: Map<string, Node> = new Map();
    private _routerMac?: string;
    public get router(): Router | undefined {
        return this._routerMac
            ? (this._nodes.get(this._routerMac) as Router)
            : undefined;
    }
    public get nodes(): Node[] {
        return [...this._nodes.values()];
    }
    public get connections(): Connection[] {
        return this.router?.connections ?? [];
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

    public addNode(type: NodeType, position?: Position) {
        if (this._routerMac && type === NodeType.ROUTER) {
            toast.error("No se puede agregar más de un router a la red.");
            throw new Error("Cannot add more than one router to the network");
        }

        this._dialog
            .open(AddNodeDialogComponent, { context: { type } })
            .closed$.subscribe((context: AddNodeDialogContext) => {
                if (!context) return;

                const node = this._addNode(context.name, type, position);

                toast.success(`Se ha añadido ${context.name} correctamente.`);
                console.log("Added new node to the network", node);
            });
    }

    public deleteNode(mac: string) {
        if (!this.exists(mac)) {
            toast.error("No existe un nodo con la dirección MAC especificada.");
            throw new Error(
                "Does not exist a node with the specified MAC address",
            );
        }

        this._dialog
            .open(DeleteNodeDialogComponent, {
                context: { node: this.findByMac(mac) },
            })
            .closed$.subscribe((context: DeleteNodeDialogContext) => {
                if (!context) return;

                this._deleteNode(context.node.mac);
                toast.success(
                    `Se ha eliminado ${context.node.name} correctamente.`,
                );
            });
    }
}
