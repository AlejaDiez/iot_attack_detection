import { inject } from "@angular/core";
import {
    ActivatedRouteSnapshot,
    CanActivateFn,
    GuardResult,
    RedirectCommand,
    Router,
    RouterStateSnapshot,
} from "@angular/router";
import { NodeType } from "@models/node";
import { NetworkManagerService } from "@services/network-manager.service";

/**
 * Protector que verifica si un nodo es de tipo "atacante".
 *
 * @param route La ruta activada.
 * @param state El estado del router.
 * @returns `true` si el nodo es un atacante, `false` en caso contrario.
 */
export const attackingNodeGuard: CanActivateFn = (
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot,
): GuardResult => {
    const router = inject(Router);
    const networkManager = inject(NetworkManagerService);
    const { mac } = route.parent!.params;
    const node = networkManager.findByMac(mac);

    return NodeType.AttackerTypes.includes(node.type)
        ? true
        : new RedirectCommand(router.parseUrl(`/${mac}/network-traffic`));
};
