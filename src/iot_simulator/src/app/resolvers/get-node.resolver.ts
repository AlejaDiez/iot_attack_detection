import { inject } from "@angular/core";
import { ResolveFn } from "@angular/router";
import { Node } from "@models/node";
import { NetworkManagerService } from "@services/network-manager.service";

/**
 * Resolutor que obtiene un nodo de la red por su dirección MAC.
 *
 * @param route La ruta activada.
 * @param state El estado del router.
 * @returns Nodo correspondiente a la dirección MAC proporcionada.
 */
export const getNodeResolver: ResolveFn<Node> = (route, state): Node => {
    const networkManager = inject(NetworkManagerService);
    let mac = route.params["mac"];

    if (!mac) mac = route.parent!.params["mac"];
    return networkManager.findByMac(mac);
};
