import { inject } from "@angular/core";
import {
    ActivatedRouteSnapshot,
    CanActivateFn,
    RouterStateSnapshot,
} from "@angular/router";
import { NetworkManagerService } from "@services/network-manager.service";

/**
 * Protector que verifica si un nodo existe en la red.
 *
 * @param route La ruta activada.
 * @param state El estado del router.
 * @returns `true` si el nodo existe, `false` en caso contrario.
 */
export const nodeExistsGuard: CanActivateFn = (
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot,
) => {
    const networkManager = inject(NetworkManagerService);
    const { mac } = route.params;

    return networkManager.exists(mac);
};
