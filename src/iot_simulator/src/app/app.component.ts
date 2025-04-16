import { Component, HostListener, ViewChild } from "@angular/core";
import { Router as NavigationRouter, RouterOutlet } from "@angular/router";
import { CanvasComponent } from "@components/canvas/canvas.component";
import { ConnectionComponent } from "@components/connection/connection.component";
import { MenuBarComponent } from "@components/menu-bar/menu-bar.component";
import { NodeComponent } from "@components/node/node.component";
import { HlmButtonModule } from "@components/ui/ui-button-helm/src";
import { Connection } from "@models/connection";
import { Node, NodeType } from "@models/node";
import { Router } from "@models/router";
import { NgIcon, provideIcons } from "@ng-icons/core";
import {
    lucideChevronsLeftRightEllipsis,
    lucideGhost,
    lucideSettings,
    lucideTrash,
} from "@ng-icons/lucide";
import { ConfigService } from "@services/config.service";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnContextMenuTriggerDirective } from "@spartan-ng/brain/menu";
import {
    HlmMenuComponent,
    HlmMenuGroupComponent,
    HlmMenuItemDirective,
    HlmMenuItemIconDirective,
    HlmMenuSeparatorComponent,
    HlmMenuShortcutComponent,
} from "@spartan-ng/ui-menu-helm";
import { HlmToasterComponent } from "@spartan-ng/ui-sonner-helm";
import { floatAnimation } from "./app.routes.transition";

@Component({ template: "" })
export class BlankComponent {}

@Component({
    selector: "app-root",
    imports: [
        RouterOutlet,
        BrnContextMenuTriggerDirective,
        CanvasComponent,
        ConnectionComponent,
        HlmMenuComponent,
        HlmMenuGroupComponent,
        HlmMenuItemDirective,
        HlmMenuItemIconDirective,
        HlmMenuSeparatorComponent,
        HlmMenuShortcutComponent,
        HlmToasterComponent,
        MenuBarComponent,
        NodeComponent,
        NgIcon,

        HlmButtonModule,
    ],
    providers: [
        provideIcons({
            lucideChevronsLeftRightEllipsis,
            lucideGhost,
            lucideSettings,
            lucideTrash,
        }),
    ],
    templateUrl: "app.component.html",
    animations: [floatAnimation],
})
export class AppComponent {
    @ViewChild(CanvasComponent)
    protected readonly canvas!: CanvasComponent;
    protected get nodes(): Node[] {
        return this._networkManager.nodes;
    }
    protected get connections(): Connection[] {
        return this._networkManager.router?.connections ?? [];
    }
    protected get router(): Router | undefined {
        return this._networkManager.router;
    }

    public constructor(
        private readonly _router: NavigationRouter,
        private readonly _config: ConfigService,
        private readonly _networkManager: NetworkManagerService,
    ) {}

    protected getAnimationData(outlet: RouterOutlet) {
        return (
            (outlet &&
                outlet.isActivated &&
                outlet.activatedRoute.snapshot.params["mac"]) ||
            "_"
        );
    }

    @HostListener("document:keydown.meta.n", ["$event"])
    protected onNewFile(event?: Event) {
        event?.preventDefault();
        this._networkManager.new();
    }

    @HostListener("document:keydown.meta.o", ["$event"])
    protected onOpenFile(event?: Event) {
        event?.preventDefault();
        this._networkManager.loadFromFile();
    }

    protected onLoadExternalLibrary(event?: Event) {
        event?.preventDefault();
        this._config.libraryManager.loadFromFile();
    }

    @HostListener("document:keydown.meta.shift.s", ["$event"])
    protected onSaveFile(event?: Event) {
        event?.preventDefault();
        this._networkManager.saveToFile();
    }

    @HostListener("document:keydown.meta.z", ["$event"])
    protected onUndo(event?: Event) {
        event?.preventDefault();
        this._config.stateManager.undo();
    }

    @HostListener("document:keydown.meta.shift.z", ["$event"])
    protected onRedo(event?: Event) {
        event?.preventDefault();
        this._config.stateManager.redo();
    }

    @HostListener("document:keydown.meta.shift.r", ["'router'", "$event"])
    @HostListener("document:keydown.meta.shift.d", ["undefined", "$event"])
    protected onInsertNode(type?: string, event?: Event) {
        const { scrollWidth, scrollHeight, scrollLeft, scrollTop } =
            this.canvas.scrollable.nativeElement;
        const { left, top, width, height } =
            this.canvas.scrollable.nativeElement.getBoundingClientRect();
        const centerX = scrollWidth / 2;
        const centerY = scrollHeight / 2;

        event?.preventDefault();
        if (event instanceof MouseEvent) {
            const { clientX, clientY } = event;

            this._networkManager.addNode(type as NodeType, {
                x: clientX - left + scrollLeft - centerX,
                y: centerY - (clientY - top + scrollTop),
            });
        } else {
            this._networkManager.addNode(type as NodeType, {
                x: width / 2 - left + scrollLeft - centerX,
                y: centerY - (height / 2 - top + scrollTop),
            });
        }
    }

    protected onDeleteNode(mac: string, event?: Event) {
        event?.preventDefault();
        this._networkManager.deleteNode(mac).then((value) => {
            const { url } = this._router;

            if (value && url.includes(mac)) this._router.navigate([""]);
        });
    }

    protected onNodeTraffic(mac: string, event?: Event) {
        event?.preventDefault();
        this._router.navigate([mac, "network-traffic"]);
    }

    protected onNodeAttack(mac: string, event?: Event) {
        event?.preventDefault();
        this._router.navigate([mac, "attack"]);
    }

    protected onNodeConfig(mac: string, event?: Event) {
        event?.preventDefault();
        this._router.navigate([mac, "configuration"]);
    }
}
