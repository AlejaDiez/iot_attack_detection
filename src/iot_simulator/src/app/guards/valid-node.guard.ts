import { inject } from "@angular/core";
import {
    ActivatedRouteSnapshot,
    GuardResult,
    MaybeAsync,
    Router,
    RouterStateSnapshot,
} from "@angular/router";
import { NetworkManagerService } from "@services/network-manager.service";

export const validNodeGuard = (
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot,
    ...args: any[]
): MaybeAsync<GuardResult> => {
    const router = inject(Router);
    const networkManager = inject(NetworkManagerService);
    const { mac } = route.parent!.params;

    if (networkManager.exists(mac)) {
        const node = networkManager.findByMac(mac);

        if (args.length === 0 || args.includes(node.type)) return true;
    }
    router.navigateByUrl("/");
    return false;
};
